#!/usr/bin/env python3
"""Serve a blinded ACL60/60 frame-only validator workspace."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import threading
from urllib.parse import parse_qs, urlparse

from scripts.acl6060_source_event_annotation_v2 import (
    GLOBAL_FORBIDDEN_KEY_PARTS,
    assert_no_forbidden_keys,
    canonical_hash,
    frame_task_lock_payload,
    load_json,
    load_jsonl,
    validate_frame_annotation,
)
from scripts.serve_acl6060_authoring import (
    clean_string_list,
    clean_text,
    utc_now,
    write_jsonl_atomic,
)


EDITABLE_FIELDS = {
    "frame_support_status",
    "frame_answer_option_id",
    "evidence_subtypes",
    "frame_confidence",
    "frame_note",
    "frame_submitted_at_utc",
    "frame_annotation_lock_sha256",
}


class FrameValidationSession:
    def __init__(
        self,
        *,
        input_sheet: Path,
        workspace_root: Path,
        working_sheet: Path,
        config_path: Path,
    ) -> None:
        if input_sheet.resolve() == working_sheet.resolve():
            raise ValueError("Working sheet must not overwrite the input sheet")
        if working_sheet.is_symlink():
            raise ValueError("Working sheet cannot be a symlink")
        self.workspace_root = workspace_root.resolve(strict=True)
        self.working_sheet = working_sheet
        self.config = load_json(config_path)
        self.lock = threading.Lock()
        source_rows = load_jsonl(input_sheet)
        if not source_rows:
            raise ValueError("Frame-validation input sheet is empty")
        self._validate_source_rows(source_rows)
        if working_sheet.exists():
            rows = load_jsonl(working_sheet)
            self._validate_resume_rows(source_rows, rows)
        else:
            rows = source_rows
            write_jsonl_atomic(working_sheet, rows)
        os.chmod(working_sheet, 0o600)
        self.rows = rows
        self._frame_paths = [self._resolve_frame(row) for row in rows]

    @staticmethod
    def _immutable_payload(row: dict) -> dict:
        return {key: value for key, value in row.items() if key not in EDITABLE_FIELDS}

    def _validate_source_rows(self, rows: list[dict]) -> None:
        packet_ids = [row.get("packet_id") for row in rows]
        if len(packet_ids) != len(set(packet_ids)):
            raise ValueError("Frame-validation input contains duplicate packet ids")
        validator_ids = {row.get("validator_id") for row in rows}
        if len(validator_ids) != 1 or None in validator_ids:
            raise ValueError("Frame-validation input must belong to one validator")
        for row in rows:
            assert_no_forbidden_keys(
                row, GLOBAL_FORBIDDEN_KEY_PARTS + ("audio", "canonical")
            )
            if row.get("stage") != "validator_frame_only_answer_pass":
                raise ValueError("Frame-validation input contains a non-frame stage")
            if row.get("frame_support_status") != "pending":
                raise ValueError("Frame-validation input must be an untouched pending sheet")
            if row.get("frame_annotation_lock_sha256") is not None:
                raise ValueError("Frame-validation input is already frozen")
            if row.get("frame_task_lock_sha256") != canonical_hash(
                frame_task_lock_payload(row)
            ):
                raise ValueError(f'Frame task lock mismatch: {row.get("packet_id")}')
            self._resolve_frame(row)

    def _validate_resume_rows(
        self, source_rows: list[dict], rows: list[dict]
    ) -> None:
        if len(rows) != len(source_rows):
            raise ValueError("Working sheet row count differs from the input sheet")
        for source, row in zip(source_rows, rows, strict=True):
            if self._immutable_payload(source) != self._immutable_payload(row):
                raise ValueError("Working sheet changed an immutable frame task field")
            assert_no_forbidden_keys(
                row, GLOBAL_FORBIDDEN_KEY_PARTS + ("audio", "canonical")
            )
            if row.get("frame_annotation_lock_sha256") is not None:
                raise ValueError("Working sheet is already frozen")
            if row.get("frame_support_status") != "pending":
                validate_frame_annotation(row, self.config)

    def _resolve_frame(self, row: dict) -> Path:
        relative = PurePosixPath(str(row.get("frame_path", "")))
        if relative.is_absolute() or ".." in relative.parts or str(relative) != row.get(
            "frame_path"
        ):
            raise ValueError("Frame path must be canonical and relative")
        path = self.workspace_root.joinpath(*relative.parts)
        component = self.workspace_root
        for part in relative.parts:
            component = component / part
            if component.is_symlink():
                raise ValueError("Frame path cannot traverse a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(self.workspace_root):
            raise ValueError("Frame path escapes the workspace root")
        return resolved

    def state(self, index: int) -> dict:
        if not 0 <= index < len(self.rows):
            raise ValueError("Frame-validation item index is out of range")
        row = self.rows[index]
        return {
            "index": index,
            "total_count": len(self.rows),
            "completed_count": sum(
                item["frame_support_status"] != "pending" for item in self.rows
            ),
            "packet_id": row["packet_id"],
            "source_question": row["source_question"],
            "source_options": row["source_options"],
            "frame_support_status": row["frame_support_status"],
            "frame_answer_option_id": row["frame_answer_option_id"],
            "evidence_subtypes": row["evidence_subtypes"],
            "frame_confidence": row["frame_confidence"],
            "frame_note": row["frame_note"],
            "allowed_statuses": self.config["frame_validation"][
                "frame_support_status"
            ],
            "allowed_evidence_subtypes": self.config["evidence_subtypes"],
        }

    def frame_path(self, index: int) -> Path:
        if not 0 <= index < len(self._frame_paths):
            raise ValueError("Frame-validation item index is out of range")
        return self._frame_paths[index]

    def save(self, payload: dict) -> dict:
        with self.lock:
            index = payload.get("index")
            if not isinstance(index, int) or not 0 <= index < len(self.rows):
                raise ValueError("Frame-validation item index is invalid")
            row = self.rows[index]
            if payload.get("packet_id") != row["packet_id"]:
                raise ValueError("Frame-validation payload targets the wrong packet")
            status = payload.get("frame_support_status")
            allowed = self.config["frame_validation"]["frame_support_status"]
            if status not in allowed or status == "pending":
                raise ValueError("Frame support status is invalid")
            confidence = payload.get("frame_confidence")
            if confidence not in {"low", "medium", "high"}:
                raise ValueError("Frame confidence is invalid")

            answer = payload.get("frame_answer_option_id")
            evidence_subtypes = clean_string_list(
                payload.get("evidence_subtypes", [])
            )
            if status != "supported":
                answer = None
                evidence_subtypes = []

            updated = {
                **row,
                "frame_support_status": status,
                "frame_answer_option_id": answer,
                "evidence_subtypes": evidence_subtypes,
                "frame_confidence": confidence,
                "frame_note": clean_text(payload.get("frame_note", "")),
                "frame_submitted_at_utc": utc_now(),
                "frame_annotation_lock_sha256": None,
            }
            validate_frame_annotation(updated, self.config)
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
  <title>ACL Frame Validation</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f6f7; color: #1c2024; }
    button, textarea { font: inherit; }
    button { cursor: pointer; }
    header { height: 56px; padding: 0 20px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #d5dadd; background: #fff; }
    header strong { white-space: nowrap; }
    .progress, .packet { color: #5f6872; font-variant-numeric: tabular-nums; }
    .spacer { flex: 1; }
    .nav { width: 36px; height: 36px; border: 1px solid #b7bdc3; border-radius: 6px; background: #fff; font-weight: 700; }
    main { display: grid; grid-template-columns: minmax(440px, 1.22fr) minmax(380px, .78fr); min-height: calc(100vh - 56px); }
    .visual { padding: 20px; background: #22272c; display: flex; align-items: center; justify-content: center; min-width: 0; }
    .visual img { display: block; width: 100%; max-height: calc(100vh - 96px); object-fit: contain; }
    .form { padding: 24px 26px 40px; overflow: auto; background: #fff; border-left: 1px solid #d5dadd; }
    .meta { display: flex; justify-content: space-between; margin-bottom: 18px; font-size: 13px; }
    .question { margin: 0 0 20px; font-size: 20px; line-height: 1.35; }
    fieldset { border: 0; padding: 0; margin: 0 0 20px; }
    legend, label.title { display: block; margin-bottom: 8px; font-size: 13px; font-weight: 700; }
    .choices { display: grid; gap: 8px; }
    .choice { display: flex; align-items: flex-start; gap: 9px; padding: 11px 12px; border: 1px solid #c4c9ce; border-radius: 6px; background: #fff; }
    .choice:has(input:checked) { border-color: #1769aa; background: #edf5fb; }
    .segmented { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
    .segment { position: relative; }
    .segment input { position: absolute; opacity: 0; pointer-events: none; }
    .segment span { display: block; padding: 9px 6px; border: 1px solid #b7bdc3; border-radius: 5px; text-align: center; font-size: 13px; }
    .segment input:checked + span { border-color: #1769aa; background: #1769aa; color: #fff; font-weight: 700; }
    .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
    .check { display: flex; align-items: flex-start; gap: 7px; font-size: 13px; }
    textarea { width: 100%; min-height: 80px; resize: vertical; border: 1px solid #aeb5bc; border-radius: 5px; padding: 9px 10px; color: #1c2024; }
    .actions { position: sticky; bottom: -40px; padding: 14px 0 2px; background: #fff; }
    .save { width: 100%; min-height: 42px; border: 1px solid #1769aa; border-radius: 5px; background: #1769aa; color: #fff; font-weight: 700; }
    .toast { min-height: 20px; margin-top: 8px; color: #aa3517; font-size: 13px; }
    .hidden { display: none; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .visual { min-height: 42vh; }
      .visual img { max-height: 58vh; }
      .form { border-left: 0; border-top: 1px solid #d5dadd; }
    }
    @media (max-width: 540px) {
      header { padding: 0 10px; gap: 8px; }
      header strong { font-size: 14px; }
      .wide-title { display: none; }
      .nav { width: 32px; height: 32px; }
      .form { padding: 20px 22px 36px; }
      .question { font-size: 18px; }
      .checks { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <strong>ACL <span class="wide-title">Frame </span>Validation</strong>
    <span class="progress" id="progress"></span>
    <span class="spacer"></span>
    <button class="nav" id="prev" title="Previous item" aria-label="Previous item">&lt;</button>
    <button class="nav" id="next" title="Next item" aria-label="Next item">&gt;</button>
  </header>
  <main>
    <section class="visual"><img id="frame" alt="Current slide frame"></section>
    <section class="form">
      <div class="meta"><span class="packet" id="packet"></span><span id="statusText"></span></div>
      <h1 class="question" id="question"></h1>
      <fieldset><legend>Does the frame support a unique answer?</legend><div class="segmented" id="statuses"></div></fieldset>
      <fieldset id="answerFields"><legend>Answer</legend><div class="choices" id="options"></div></fieldset>
      <fieldset id="subtypeFields"><legend>Evidence subtype</legend><div class="checks" id="subtypes"></div></fieldset>
      <fieldset><legend>Confidence</legend><div class="segmented" id="confidence"></div></fieldset>
      <label class="title" for="note">Validator note</label><textarea id="note"></textarea>
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
const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function selected(name) {
  const node = document.querySelector(`input[name="${name}"]:checked`);
  return node ? node.value : null;
}

function renderSegments(root, name, values, current) {
  root.innerHTML = "";
  values.forEach((value) => {
    const label = document.createElement("label");
    label.className = "segment";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = value;
    input.checked = value === current;
    input.onchange = renderSupportFields;
    const text = document.createElement("span");
    text.textContent = value;
    label.append(input, text);
    root.appendChild(label);
  });
}

function renderOptions(options, current) {
  const root = $("#options");
  root.innerHTML = "";
  options.forEach((option) => {
    const label = document.createElement("label");
    label.className = "choice";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "answer";
    input.value = option.option_id;
    input.checked = option.option_id === current;
    label.append(input, document.createTextNode(option.text));
    root.appendChild(label);
  });
}

function renderChecks(values, current) {
  const root = $("#subtypes");
  root.innerHTML = "";
  values.forEach((value) => {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    input.checked = current.includes(value);
    label.append(input, document.createTextNode(value.replaceAll("_", " ")));
    root.appendChild(label);
  });
}

function renderSupportFields() {
  const supported = selected("support") === "supported";
  $("#answerFields").classList.toggle("hidden", !supported);
  $("#subtypeFields").classList.toggle("hidden", !supported);
}

function setIndex(index) {
  currentIndex = Math.max(0, Math.min(item.total_count - 1, index));
  history.replaceState(null, "", `?index=${currentIndex}`);
  load();
}

async function load() {
  item = await api(`/api/item?index=${currentIndex}`);
  $("#progress").textContent = `${item.completed_count}/${item.total_count}`;
  $("#packet").textContent = item.packet_id;
  $("#statusText").textContent = item.frame_support_status.replaceAll("_", " ");
  $("#question").textContent = item.source_question;
  $("#frame").src = `/api/frame?index=${currentIndex}&v=${encodeURIComponent(item.packet_id)}`;
  $("#prev").disabled = currentIndex === 0;
  $("#next").disabled = currentIndex + 1 === item.total_count;
  const support = item.frame_support_status === "pending" ? "supported" : item.frame_support_status;
  renderSegments($("#statuses"), "support", item.allowed_statuses.filter((value) => value !== "pending"), support);
  renderOptions(item.source_options, item.frame_answer_option_id);
  renderChecks(item.allowed_evidence_subtypes, item.evidence_subtypes || []);
  renderSegments($("#confidence"), "confidence", ["low", "medium", "high"], item.frame_confidence || "medium");
  $("#note").value = item.frame_note || "";
  renderSupportFields();
  $("#toast").textContent = "";
}

$("#prev").onclick = () => setIndex(currentIndex - 1);
$("#next").onclick = () => setIndex(currentIndex + 1);
$("#save").onclick = async () => {
  const payload = {
    index: currentIndex,
    packet_id: item.packet_id,
    frame_support_status: selected("support"),
    frame_answer_option_id: selected("answer"),
    evidence_subtypes: [...document.querySelectorAll("#subtypes input:checked")].map((node) => node.value),
    frame_confidence: selected("confidence"),
    frame_note: $("#note").value,
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


def make_handler(session: FrameValidationSession):
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
            self.send_bytes(json.dumps(value).encode(), "application/json", status)

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
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43189)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Frame-validation service may bind only to localhost")
    session = FrameValidationSession(
        input_sheet=args.input_sheet,
        workspace_root=args.workspace_root,
        working_sheet=args.working_sheet,
        config_path=args.config,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
