#!/usr/bin/env python3
"""Serve the blinded ACL60/60 frame-only question-authoring workspace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import threading
from urllib.parse import parse_qs, urlparse
import uuid

from scripts.acl6060_source_event_annotation_v2 import (
    GLOBAL_FORBIDDEN_KEY_PARTS,
    assert_no_forbidden_keys,
    load_json,
    load_jsonl,
    validate_author_row,
)


IMMUTABLE_TASK_FIELDS = (
    "schema_version",
    "packet_id",
    "frame_path",
    "frame_binding_hmac",
    "stage",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )
    try:
        with temporary.open("x", encoding="utf-8") as output:
            os.chmod(temporary, 0o600)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def clean_text(value: object, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected text input")
    normalized = " ".join(value.strip().split())
    if required and not normalized:
        raise ValueError("Required text input is blank")
    return normalized


def clean_string_list(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Expected a list of text values")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def clean_region(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"x", "y", "w", "h"}:
        raise ValueError("Evidence region must contain x, y, w and h")
    region = {key: float(value[key]) for key in ("x", "y", "w", "h")}
    if any(not 0.0 <= region[key] <= 1.0 for key in region):
        raise ValueError("Evidence region coordinates must be normalized")
    if region["w"] <= 0.0 or region["h"] <= 0.0:
        raise ValueError("Evidence region must have positive area")
    if region["x"] + region["w"] > 1.000001:
        raise ValueError("Evidence region exceeds the frame width")
    if region["y"] + region["h"] > 1.000001:
        raise ValueError("Evidence region exceeds the frame height")
    return {key: round(value, 6) for key, value in region.items()}


class AuthoringSession:
    def __init__(
        self,
        *,
        input_sheet: Path,
        workspace_root: Path,
        working_sheet: Path,
        config_path: Path,
        author_id: str,
    ) -> None:
        if input_sheet.resolve() == working_sheet.resolve():
            raise ValueError("Working sheet must not overwrite the input sheet")
        if working_sheet.is_symlink():
            raise ValueError("Working sheet cannot be a symlink")
        if not author_id.strip() or author_id == "author_pending":
            raise ValueError("A non-placeholder author id is required")
        self.workspace_root = workspace_root.resolve(strict=True)
        self.working_sheet = working_sheet
        self.config = load_json(config_path)
        self.lock = threading.Lock()
        source_rows = load_jsonl(input_sheet)
        if not source_rows:
            raise ValueError("Authoring input sheet is empty")
        self._validate_source_rows(source_rows)
        if working_sheet.exists():
            rows = load_jsonl(working_sheet)
            self._validate_resume_rows(source_rows, rows, author_id)
        else:
            rows = [{**row, "author_id": author_id} for row in source_rows]
            write_jsonl_atomic(working_sheet, rows)
        os.chmod(working_sheet, 0o600)
        self.rows = rows
        self._frame_paths = [self._resolve_frame(row) for row in rows]

    def _validate_source_rows(self, rows: list[dict]) -> None:
        packet_ids = [row.get("packet_id") for row in rows]
        if len(packet_ids) != len(set(packet_ids)):
            raise ValueError("Authoring input contains duplicate packet ids")
        for row in rows:
            assert_no_forbidden_keys(row, GLOBAL_FORBIDDEN_KEY_PARTS + ("audio",))
            if row.get("stage") != "frame_only_question_authoring":
                raise ValueError("Authoring input contains a non-frame stage")
            if row.get("authoring_status") != "pending":
                raise ValueError("Authoring input must be an untouched pending sheet")
            self._resolve_frame(row)

    def _validate_resume_rows(
        self,
        source_rows: list[dict],
        rows: list[dict],
        author_id: str,
    ) -> None:
        if len(rows) != len(source_rows):
            raise ValueError("Working sheet row count differs from the input sheet")
        for source, row in zip(source_rows, rows, strict=True):
            if any(source[field] != row.get(field) for field in IMMUTABLE_TASK_FIELDS):
                raise ValueError("Working sheet changed an immutable authoring task field")
            if row.get("author_id") != author_id:
                raise ValueError("Working sheet belongs to a different author")
            assert_no_forbidden_keys(row, GLOBAL_FORBIDDEN_KEY_PARTS + ("audio",))
            if row.get("authoring_status") != "pending":
                validate_author_row(row, self.config)

    def _resolve_frame(self, row: dict) -> Path:
        relative = PurePosixPath(str(row.get("frame_path", "")))
        if relative.is_absolute() or ".." in relative.parts or str(relative) != row.get(
            "frame_path"
        ):
            raise ValueError("Authoring frame path must be canonical and relative")
        path = self.workspace_root.joinpath(*relative.parts)
        component = self.workspace_root
        for part in relative.parts:
            component = component / part
            if component.is_symlink():
                raise ValueError("Authoring frame cannot traverse a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(self.workspace_root):
            raise ValueError("Authoring frame escapes the workspace root")
        return resolved

    def state(self, index: int) -> dict:
        if not 0 <= index < len(self.rows):
            raise ValueError("Authoring item index is out of range")
        row = self.rows[index]
        return {
            "index": index,
            "total_count": len(self.rows),
            "completed_count": sum(
                item["authoring_status"] != "pending" for item in self.rows
            ),
            "packet_id": row["packet_id"],
            "authoring_status": row["authoring_status"],
            "source_question": row["source_question"],
            "source_options": row["source_options"],
            "canonical_answer_index": row["canonical_answer_index"],
            "evidence_subtypes": row["evidence_subtypes"],
            "evidence_region": row["evidence_region"],
            "term_or_entity": row["term_or_entity"],
            "negative_labels": row["negative_labels"],
            "exclusion_labels": row["exclusion_labels"],
            "authoring_note": row["authoring_note"],
            "allowed_statuses": self.config["question_authoring"]["authoring_status"],
            "allowed_evidence_subtypes": self.config["evidence_subtypes"],
            "allowed_negative_labels": self.config["negative_labels"],
            "allowed_exclusion_labels": self.config["exclusion_labels"],
        }

    def frame_path(self, index: int) -> Path:
        if not 0 <= index < len(self._frame_paths):
            raise ValueError("Authoring item index is out of range")
        return self._frame_paths[index]

    def save(self, payload: dict) -> dict:
        with self.lock:
            index = payload.get("index")
            if not isinstance(index, int) or not 0 <= index < len(self.rows):
                raise ValueError("Authoring item index is invalid")
            row = self.rows[index]
            if payload.get("packet_id") != row["packet_id"]:
                raise ValueError("Authoring payload targets the wrong packet")
            status = payload.get("authoring_status")
            if status not in self.config["question_authoring"]["authoring_status"]:
                raise ValueError("Authoring status is invalid")
            if status == "pending":
                raise ValueError("Save requires a completed authoring status")

            updated = dict(row)
            updated.update(
                {
                    "authoring_status": status,
                    "authoring_note": clean_text(payload.get("authoring_note", "")),
                    "authoring_completed_at_utc": utc_now(),
                    "authoring_lock_sha256": None,
                    "question_lock_sha256": None,
                }
            )
            if status == "candidate":
                options = clean_string_list(payload.get("source_options"))
                answer = payload.get("canonical_answer_index")
                if not isinstance(answer, int):
                    raise ValueError("Candidate requires a canonical answer")
                updated.update(
                    {
                        "source_question": clean_text(
                            payload.get("source_question"), required=True
                        ),
                        "source_options": options,
                        "canonical_answer_index": answer,
                        "evidence_subtypes": clean_string_list(
                            payload.get("evidence_subtypes")
                        ),
                        "evidence_region": clean_region(payload.get("evidence_region")),
                        "term_or_entity": clean_text(
                            payload.get("term_or_entity"), required=True
                        ),
                        "negative_labels": [],
                        "exclusion_labels": [],
                        "question_locked_at_utc": utc_now(),
                    }
                )
            else:
                updated.update(
                    {
                        "source_question": None,
                        "source_options": [],
                        "canonical_answer_index": None,
                        "evidence_subtypes": [],
                        "evidence_region": None,
                        "term_or_entity": None,
                        "negative_labels": (
                            clean_string_list(payload.get("negative_labels"))
                            if status == "negative_no_visual_question"
                            else []
                        ),
                        "exclusion_labels": (
                            clean_string_list(payload.get("exclusion_labels"))
                            if status == "exclude_other"
                            else []
                        ),
                        "question_locked_at_utc": None,
                    }
                )
            validate_author_row(updated, self.config)
            next_rows = list(self.rows)
            next_rows[index] = updated
            write_jsonl_atomic(self.working_sheet, next_rows)
            self.rows = next_rows
            return self.state(index)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ACL Source Event Authoring</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f6f7; color: #1c2024; }
    button, input, textarea, select { font: inherit; }
    button { cursor: pointer; }
    header { height: 56px; padding: 0 20px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #d7dadd; background: #fff; }
    header strong { white-space: nowrap; }
    .progress { color: #5f6872; font-variant-numeric: tabular-nums; }
    .spacer { flex: 1; }
    .nav { width: 36px; height: 36px; border: 1px solid #b7bdc3; border-radius: 6px; background: #fff; font-weight: 700; }
    main { display: grid; grid-template-columns: minmax(420px, 1.2fr) minmax(360px, .8fr); min-height: calc(100vh - 56px); }
    .visual { padding: 20px; background: #22272c; display: flex; align-items: center; justify-content: center; min-width: 0; }
    .frame-wrap { position: relative; width: 100%; max-width: 1100px; user-select: none; touch-action: none; }
    .frame-wrap img { display: block; width: 100%; max-height: calc(100vh - 96px); object-fit: contain; }
    .selection { position: absolute; border: 2px solid #20b486; background: rgba(32, 180, 134, .13); pointer-events: none; }
    .form { padding: 22px 24px 40px; overflow: auto; background: #fff; border-left: 1px solid #d7dadd; }
    .meta { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; color: #5f6872; font-size: 13px; }
    .status-dot { width: 9px; height: 9px; border-radius: 50%; background: #a9b0b7; display: inline-block; margin-right: 7px; }
    .status-dot.done { background: #218f68; }
    label, legend { display: block; font-size: 13px; font-weight: 650; margin-bottom: 6px; }
    input[type="text"], textarea, select { width: 100%; border: 1px solid #aeb5bc; border-radius: 5px; padding: 9px 10px; background: #fff; color: #1c2024; }
    textarea { min-height: 78px; resize: vertical; }
    .field { margin-bottom: 16px; }
    fieldset { border: 0; padding: 0; margin: 0 0 16px; }
    .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
    .check { display: flex; align-items: flex-start; gap: 7px; font-size: 13px; font-weight: 450; margin: 0; }
    .option { display: grid; grid-template-columns: 24px 1fr 32px; gap: 7px; align-items: center; margin-bottom: 7px; }
    .icon { width: 32px; height: 32px; border: 1px solid #b7bdc3; border-radius: 5px; background: #fff; font-weight: 700; }
    .option-tools { display: flex; justify-content: flex-end; margin-top: 6px; }
    .secondary { min-height: 34px; border: 1px solid #aeb5bc; border-radius: 5px; background: #fff; padding: 6px 10px; }
    .region { display: flex; align-items: center; gap: 10px; color: #5f6872; font-size: 12px; }
    .actions { position: sticky; bottom: -40px; display: flex; gap: 10px; padding: 14px 0 2px; background: #fff; }
    .save { min-height: 42px; flex: 1; border: 1px solid #1769aa; border-radius: 5px; background: #1769aa; color: #fff; font-weight: 700; }
    .toast { min-height: 20px; color: #aa3517; font-size: 13px; margin-top: 8px; }
    .hidden { display: none; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .visual { min-height: 42vh; }
      .form { border-left: 0; border-top: 1px solid #d7dadd; }
      .frame-wrap img { max-height: 58vh; }
    }
    @media (max-width: 540px) {
      header { padding: 0 10px; gap: 8px; }
      header strong { font-size: 14px; }
      .wide-title { display: none; }
      .nav { width: 32px; height: 32px; }
      .form { padding: 18px 22px 36px; }
      .checks { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <strong>ACL <span class="wide-title">Source Event </span>Authoring</strong>
    <span class="progress" id="progress"></span>
    <span class="spacer"></span>
    <button class="nav" id="prev" title="Previous item" aria-label="Previous item">&lt;</button>
    <button class="nav" id="next" title="Next item" aria-label="Next item">&gt;</button>
  </header>
  <main>
    <section class="visual">
      <div class="frame-wrap" id="frameWrap">
        <img id="frame" alt="Current slide frame" draggable="false">
        <div class="selection hidden" id="selection"></div>
      </div>
    </section>
    <section class="form">
      <div class="meta"><span id="packet"></span><span><i class="status-dot" id="statusDot"></i><span id="statusText"></span></span></div>
      <div class="field">
        <label for="status">Outcome</label>
        <select id="status"></select>
      </div>
      <div id="candidateFields">
        <div class="field"><label for="question">Source-side question</label><textarea id="question"></textarea></div>
        <fieldset><legend>Options and canonical answer</legend><div id="options"></div><div class="option-tools"><button class="icon" id="addOption" title="Add option" aria-label="Add option">+</button></div></fieldset>
        <fieldset><legend>Evidence subtype</legend><div class="checks" id="subtypes"></div></fieldset>
        <div class="field"><label for="term">Term or entity</label><input id="term" type="text"></div>
        <div class="field"><label>Evidence region</label><div class="region"><span id="regionText">Not selected</span><button class="secondary" id="clearRegion">Clear box</button></div></div>
      </div>
      <fieldset class="hidden" id="negativeFields"><legend>Negative labels</legend><div class="checks" id="negativeLabels"></div></fieldset>
      <fieldset class="hidden" id="exclusionFields"><legend>Exclusion labels</legend><div class="checks" id="exclusionLabels"></div></fieldset>
      <div class="field"><label for="note">Author note</label><textarea id="note"></textarea></div>
      <div class="actions"><button class="save" id="save">Save and continue</button></div>
      <div class="toast" id="toast"></div>
    </section>
  </main>
  <script src="/app.js"></script>
</body>
</html>"""


APP_JS = r"""
let item = null;
let currentIndex = Number(new URLSearchParams(location.search).get("index") || 0);
let region = null;
let dragStart = null;

const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function setIndex(index) {
  currentIndex = Math.max(0, Math.min(item.total_count - 1, index));
  history.replaceState(null, "", `?index=${currentIndex}`);
  load();
}

function renderChecks(root, values, selected) {
  root.innerHTML = "";
  values.forEach((value) => {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    input.checked = selected.includes(value);
    label.append(input, document.createTextNode(value.replaceAll("_", " ")));
    root.appendChild(label);
  });
}

function selectedChecks(root) {
  return [...root.querySelectorAll("input:checked")].map((node) => node.value);
}

function renderOptions(values, answerIndex) {
  const root = $("#options");
  root.innerHTML = "";
  const options = values.length ? values : ["", ""];
  options.forEach((value, index) => {
    const row = document.createElement("div");
    row.className = "option";
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "canonical";
    radio.value = String(index);
    radio.checked = index === answerIndex;
    const input = document.createElement("input");
    input.type = "text";
    input.value = value;
    input.setAttribute("aria-label", `Option ${index + 1}`);
    const remove = document.createElement("button");
    remove.className = "icon";
    remove.textContent = "x";
    remove.title = "Remove option";
    remove.setAttribute("aria-label", "Remove option");
    remove.disabled = options.length <= 2;
    remove.onclick = () => {
      const next = optionValues();
      const selected = root.querySelector('input[name="canonical"]:checked');
      const selectedIndex = selected ? Number(selected.value) : null;
      next.splice(index, 1);
      const nextAnswer = selectedIndex === null || selectedIndex === index
        ? null
        : selectedIndex > index ? selectedIndex - 1 : selectedIndex;
      renderOptions(next, nextAnswer);
    };
    row.append(radio, input, remove);
    root.appendChild(row);
  });
}

function optionValues() {
  return [...$("#options").querySelectorAll('input[type="text"]')].map((node) => node.value);
}

function renderRegion() {
  const selection = $("#selection");
  if (!region) {
    selection.classList.add("hidden");
    $("#regionText").textContent = "Not selected";
    return;
  }
  selection.classList.remove("hidden");
  selection.style.left = `${region.x * 100}%`;
  selection.style.top = `${region.y * 100}%`;
  selection.style.width = `${region.w * 100}%`;
  selection.style.height = `${region.h * 100}%`;
  $("#regionText").textContent = `${Math.round(region.x * 100)}, ${Math.round(region.y * 100)} / ${Math.round(region.w * 100)} x ${Math.round(region.h * 100)}`;
}

function renderStatusFields() {
  const status = $("#status").value;
  $("#candidateFields").classList.toggle("hidden", status !== "candidate");
  $("#negativeFields").classList.toggle("hidden", status !== "negative_no_visual_question");
  $("#exclusionFields").classList.toggle("hidden", status !== "exclude_other");
}

async function load() {
  item = await api(`/api/item?index=${currentIndex}`);
  $("#progress").textContent = `${item.completed_count}/${item.total_count}`;
  $("#packet").textContent = item.packet_id;
  $("#statusText").textContent = item.authoring_status.replaceAll("_", " ");
  $("#statusDot").classList.toggle("done", item.authoring_status !== "pending");
  $("#prev").disabled = currentIndex === 0;
  $("#next").disabled = currentIndex + 1 === item.total_count;
  $("#frame").src = `/api/frame?index=${currentIndex}&v=${encodeURIComponent(item.packet_id)}`;
  $("#status").innerHTML = item.allowed_statuses
    .filter((value) => value !== "pending")
    .map((value) => `<option value="${value}">${value.replaceAll("_", " ")}</option>`)
    .join("");
  $("#status").value = item.authoring_status === "pending" ? "candidate" : item.authoring_status;
  $("#question").value = item.source_question || "";
  renderOptions(item.source_options || [], item.canonical_answer_index);
  renderChecks($("#subtypes"), item.allowed_evidence_subtypes, item.evidence_subtypes || []);
  renderChecks($("#negativeLabels"), item.allowed_negative_labels, item.negative_labels || []);
  renderChecks($("#exclusionLabels"), item.allowed_exclusion_labels, item.exclusion_labels || []);
  $("#term").value = item.term_or_entity || "";
  $("#note").value = item.authoring_note || "";
  region = item.evidence_region;
  renderRegion();
  renderStatusFields();
  $("#toast").textContent = "";
}

function framePoint(event) {
  const rect = $("#frame").getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
}

$("#frameWrap").addEventListener("pointerdown", (event) => {
  dragStart = framePoint(event);
  region = { x: dragStart.x, y: dragStart.y, w: 0.001, h: 0.001 };
  $("#frameWrap").setPointerCapture(event.pointerId);
  renderRegion();
});

$("#frameWrap").addEventListener("pointermove", (event) => {
  if (!dragStart) return;
  const point = framePoint(event);
  region = {
    x: Math.min(dragStart.x, point.x),
    y: Math.min(dragStart.y, point.y),
    w: Math.max(0.001, Math.abs(point.x - dragStart.x)),
    h: Math.max(0.001, Math.abs(point.y - dragStart.y)),
  };
  renderRegion();
});

$("#frameWrap").addEventListener("pointerup", () => { dragStart = null; });
$("#clearRegion").onclick = () => { region = null; renderRegion(); };
$("#status").onchange = renderStatusFields;
$("#prev").onclick = () => setIndex(currentIndex - 1);
$("#next").onclick = () => setIndex(currentIndex + 1);
$("#addOption").onclick = () => {
  const values = optionValues();
  const selected = $("#options").querySelector('input[name="canonical"]:checked');
  if (values.length < 4) {
    renderOptions([...values, ""], selected ? Number(selected.value) : null);
  }
};

$("#save").onclick = async () => {
  const canonical = $("#options").querySelector('input[name="canonical"]:checked');
  const payload = {
    index: currentIndex,
    packet_id: item.packet_id,
    authoring_status: $("#status").value,
    source_question: $("#question").value,
    source_options: optionValues(),
    canonical_answer_index: canonical ? Number(canonical.value) : null,
    evidence_subtypes: selectedChecks($("#subtypes")),
    evidence_region: region,
    term_or_entity: $("#term").value,
    negative_labels: selectedChecks($("#negativeLabels")),
    exclusion_labels: selectedChecks($("#exclusionLabels")),
    authoring_note: $("#note").value,
  };
  try {
    await api("/api/item", { method: "POST", body: JSON.stringify(payload) });
    if (currentIndex + 1 < item.total_count) setIndex(currentIndex + 1);
    else load();
  } catch (error) {
    $("#toast").textContent = error.message;
  }
};

load().catch((error) => { $("#toast").textContent = error.message; });
"""


def make_handler(session: AuthoringSession):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: dict, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(value).encode("utf-8"),
                "application/json",
                status,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                elif parsed.path == "/app.js":
                    self.send_bytes(APP_JS.encode(), "text/javascript; charset=utf-8")
                elif parsed.path == "/api/item":
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self.send_json(session.state(index))
                elif parsed.path == "/api/frame":
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self.send_bytes(session.frame_path(index).read_bytes(), "image/jpeg")
                else:
                    self.send_json({"error": "Not found"}, 404)
            except (ValueError, KeyError, FileNotFoundError) as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("Request body is too large")
                payload = json.loads(self.rfile.read(length))
                if self.path != "/api/item":
                    self.send_json({"error": "Not found"}, 404)
                    return
                self.send_json(session.save(payload))
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sheet", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--working-sheet", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--author-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43187)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Authoring service may bind only to localhost")
    session = AuthoringSession(
        input_sheet=args.input_sheet,
        workspace_root=args.workspace_root,
        working_sheet=args.working_sheet,
        config_path=args.config,
        author_id=args.author_id,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
