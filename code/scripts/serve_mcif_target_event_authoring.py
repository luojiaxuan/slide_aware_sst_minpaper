#!/usr/bin/env python3
"""Serve the local MCIF En-to-Zh target-event authoring workspace."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path, PurePosixPath
import threading
from urllib.parse import parse_qs, urlparse

from PIL import Image

from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256, load_jsonl
from scripts.mcif_target_event_annotation import (
    initialize_working_rows,
    load_frozen_config,
    validate_working_row,
    validate_working_rows,
    write_jsonl_atomic,
)


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected text input")
    return value.strip()


def clean_lines(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Expected a list of text values")
    cleaned = [item.strip() for item in value if item.strip()]
    return list(dict.fromkeys(cleaned))


class TargetEventAuthoringSession:
    def __init__(
        self,
        *,
        input_sheet: Path,
        expected_input_sha256: str,
        workspace_root: Path,
        working_sheet: Path,
        annotator_id: str,
        config_path: Path,
        expected_config_sha256: str,
        expected_items: int,
    ) -> None:
        if input_sheet.resolve() == working_sheet.resolve():
            raise ValueError("Working sheet must not overwrite the input sheet")
        if working_sheet.is_symlink():
            raise ValueError("Working sheet cannot be a symlink")
        if not annotator_id.strip():
            raise ValueError("A non-empty target-event annotator id is required")
        if file_sha256(input_sheet) != expected_input_sha256:
            raise ValueError("MCIF author input sheet hash differs from contract")
        if workspace_root.is_symlink():
            raise ValueError("MCIF author workspace root cannot be a symlink")
        self.workspace_root = workspace_root.resolve(strict=True)
        self.working_sheet = working_sheet
        self.annotator_id = annotator_id
        self.expected_items = expected_items
        self.config = load_frozen_config(config_path, expected_config_sha256)
        self.lock = threading.Lock()
        self.source_rows = load_jsonl(input_sheet)
        if working_sheet.exists():
            rows = load_jsonl(working_sheet)
            validate_working_rows(
                rows,
                self.source_rows,
                annotator_id=annotator_id,
                expected_items=expected_items,
                allow_pending=True,
            )
        else:
            rows = initialize_working_rows(
                self.source_rows,
                annotator_id=annotator_id,
                expected_items=expected_items,
            )
            write_jsonl_atomic(working_sheet, rows)
        self.rows = rows
        self._media_paths = [self._resolve_media(row) for row in self.source_rows]

    def _resolve_media(self, row: dict) -> Path:
        media = row["current_slide"]
        relative = PurePosixPath(str(media.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or str(relative) != media.get(
            "path"
        ):
            raise ValueError("MCIF author media path must be canonical and relative")
        cursor = self.workspace_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("MCIF author media cannot traverse a symlink")
        resolved = cursor.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(self.workspace_root):
            raise ValueError("MCIF author media escapes the workspace root")
        if file_sha256(resolved) != media["sha256"]:
            raise ValueError("MCIF author media hash differs from contract")
        with Image.open(resolved) as image:
            image.verify()
        with Image.open(resolved) as image:
            size = image.size
        if size != (media["width"], media["height"]):
            raise ValueError("MCIF author media dimensions differ from contract")
        return resolved

    def state(self, index: int) -> dict:
        if not 0 <= index < len(self.rows):
            raise ValueError("MCIF author item index is out of range")
        row = self.rows[index]
        return {
            "index": index,
            "total_count": len(self.rows),
            "completed_count": sum(
                item["annotation_status"] != "pending" for item in self.rows
            ),
            "item_id": row["item_id"],
            "source_reference_en": row["source_reference_en"],
            "target_reference_zh": row["target_reference_zh"],
            "current_slide_r0_text": row["current_slide_r0_text"],
            "current_slide_r1_text": row["current_slide_r1_text"],
            "candidate_options": row["candidate_options"],
            "annotation_status": row["annotation_status"],
            "selected_option_id": row["selected_option_id"],
            "canonical_source_event_en": row["canonical_source_event_en"],
            "acceptable_target_realizations_zh": row[
                "acceptable_target_realizations_zh"
            ],
            "forbidden_target_realizations_zh": row[
                "forbidden_target_realizations_zh"
            ],
            "target_reference_alignment": row["target_reference_alignment"],
            "slide_evidence_status": row["slide_evidence_status"],
            "annotation_note": row["annotation_note"],
            "allowed_statuses": [
                value for value in self.config["annotation_statuses"] if value != "pending"
            ],
            "allowed_target_alignments": self.config["target_reference_alignments"],
            "allowed_slide_evidence_statuses": self.config[
                "slide_evidence_statuses"
            ],
        }

    def media_path(self, index: int) -> Path:
        if not 0 <= index < len(self._media_paths):
            raise ValueError("MCIF author item index is out of range")
        return self._media_paths[index]

    def save(self, payload: dict) -> dict:
        with self.lock:
            index = payload.get("index")
            if not isinstance(index, int) or not 0 <= index < len(self.rows):
                raise ValueError("MCIF author item index is invalid")
            source = self.source_rows[index]
            current = self.rows[index]
            if payload.get("item_id") != current["item_id"]:
                raise ValueError("MCIF author payload targets the wrong item")
            status = payload.get("annotation_status")
            if status == "pending":
                raise ValueError("MCIF author save requires a completed status")
            updated = dict(current)
            updated.update(
                {
                    "annotation_status": status,
                    "target_reference_alignment": payload.get(
                        "target_reference_alignment"
                    ),
                    "slide_evidence_status": payload.get("slide_evidence_status"),
                    "annotation_note": clean_text(payload.get("annotation_note", "")),
                }
            )
            if status == "eligible":
                updated.update(
                    {
                        "selected_option_id": payload.get("selected_option_id"),
                        "canonical_source_event_en": clean_text(
                            payload.get("canonical_source_event_en", "")
                        ),
                        "acceptable_target_realizations_zh": clean_lines(
                            payload.get("acceptable_target_realizations_zh", [])
                        ),
                        "forbidden_target_realizations_zh": clean_lines(
                            payload.get("forbidden_target_realizations_zh", [])
                        ),
                    }
                )
            else:
                updated.update(
                    {
                        "selected_option_id": None,
                        "canonical_source_event_en": "",
                        "acceptable_target_realizations_zh": [],
                        "forbidden_target_realizations_zh": [],
                    }
                )
            updated["row_sha256"] = canonical_sha256(
                {key: value for key, value in updated.items() if key != "row_sha256"}
            )
            validate_working_row(
                updated,
                source,
                annotator_id=self.annotator_id,
                allow_pending=False,
            )
            next_rows = list(self.rows)
            next_rows[index] = updated
            write_jsonl_atomic(self.working_sheet, next_rows)
            self.rows = next_rows
            return self.state(index)

    def next_pending(self, current_index: int) -> int:
        for offset in range(1, len(self.rows) + 1):
            index = (current_index + offset) % len(self.rows)
            if self.rows[index]["annotation_status"] == "pending":
                return index
        return min(current_index + 1, len(self.rows) - 1)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MCIF Target Events</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f5f6; color: #182026; letter-spacing: 0; }
    button, input, textarea, select { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    header { height: 56px; padding: 0 18px; display: flex; align-items: center; gap: 14px; background: #fff; border-bottom: 1px solid #d8dcdf; }
    header strong { white-space: nowrap; }
    .progress { color: #64707a; font-variant-numeric: tabular-nums; }
    .spacer { flex: 1; }
    .nav { width: 36px; height: 36px; border: 1px solid #b8c0c6; border-radius: 6px; background: #fff; font-size: 20px; line-height: 1; }
    main { display: grid; grid-template-columns: minmax(480px, 1.1fr) minmax(440px, .9fr); min-height: calc(100vh - 56px); }
    .evidence { padding: 18px 20px 28px; overflow: auto; min-width: 0; background: #20262a; }
    .image-stage { min-height: 360px; display: flex; align-items: center; justify-content: center; }
    .image-stage img { display: block; width: 100%; max-height: 58vh; object-fit: contain; }
    .source-band, .target-band, .evidence-band { padding: 14px 0; border-top: 1px solid #465057; color: #eef1f3; }
    .target-band { color: #d9f3e9; }
    .band-label { display: block; margin-bottom: 6px; color: #9eabb3; font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .band-text { margin: 0; font-size: 15px; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; }
    details { padding-top: 10px; border-top: 1px solid #465057; }
    summary { cursor: pointer; color: #bdc7cd; font-size: 13px; }
    details pre { white-space: pre-wrap; overflow-wrap: anywhere; color: #dce2e5; font: 12px/1.45 ui-monospace, SFMono-Regular, monospace; }
    .editor { padding: 20px 22px 36px; overflow: auto; background: #fff; border-left: 1px solid #d8dcdf; }
    .meta { display: flex; justify-content: space-between; gap: 16px; color: #64707a; font-size: 13px; margin-bottom: 16px; }
    .done { color: #167554; font-weight: 700; }
    fieldset { border: 0; padding: 0; margin: 0 0 17px; min-width: 0; }
    legend, label { display: block; margin-bottom: 7px; font-size: 13px; font-weight: 700; }
    textarea, select { width: 100%; border: 1px solid #aeb7be; border-radius: 5px; padding: 9px 10px; background: #fff; color: #182026; }
    textarea { min-height: 72px; resize: vertical; }
    .segments { display: flex; flex-wrap: wrap; gap: 6px; }
    .segment input { position: absolute; opacity: 0; pointer-events: none; }
    .segment span { display: inline-flex; min-height: 34px; align-items: center; padding: 6px 9px; border: 1px solid #b8c0c6; border-radius: 5px; background: #fff; color: #46515a; font-size: 12px; }
    .segment input:checked + span { border-color: #1769aa; background: #e9f3fb; color: #0b568f; font-weight: 700; }
    .option-list { display: grid; gap: 7px; }
    .option { display: grid; grid-template-columns: 20px minmax(0, 1fr) auto; gap: 9px; align-items: start; padding: 10px; border: 1px solid #d1d6da; border-radius: 6px; background: #fafbfb; }
    .option strong { display: block; overflow-wrap: anywhere; }
    .option small { display: block; color: #6a757d; margin-top: 3px; }
    .lead { white-space: nowrap; color: #1769aa; font-variant-numeric: tabular-nums; font-size: 12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .field { margin-bottom: 17px; }
    .actions { position: sticky; bottom: -36px; padding-top: 12px; background: #fff; }
    .save { width: 100%; min-height: 42px; border: 1px solid #1769aa; border-radius: 5px; background: #1769aa; color: #fff; font-weight: 750; }
    .toast { min-height: 20px; margin-top: 8px; color: #aa3517; font-size: 13px; }
    .hidden { display: none; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .image-stage { min-height: 280px; }
      .image-stage img { max-height: 54vh; }
      .editor { border-left: 0; border-top: 1px solid #d8dcdf; }
    }
    @media (max-width: 560px) {
      header { padding: 0 10px; gap: 8px; }
      header strong { font-size: 14px; }
      .nav { width: 32px; height: 32px; }
      .evidence, .editor { padding-left: 14px; padding-right: 14px; }
      .grid { grid-template-columns: 1fr; gap: 0; }
    }
  </style>
</head>
<body>
  <header>
    <strong>MCIF En→Zh Events</strong>
    <span class="progress" id="progress"></span>
    <span class="spacer"></span>
    <button class="nav" id="prev" title="Previous item" aria-label="Previous item">‹</button>
    <button class="nav" id="next" title="Next item" aria-label="Next item">›</button>
  </header>
  <main>
    <section class="evidence">
      <div class="image-stage"><img id="slide" alt="Current slide"></div>
      <div class="source-band"><span class="band-label">English source</span><p class="band-text" id="source"></p></div>
      <div class="target-band"><span class="band-label">Chinese reference</span><p class="band-text" id="target"></p></div>
      <details><summary>R0 OCR</summary><pre id="r0"></pre></details>
      <details><summary>R1 structured text</summary><pre id="r1"></pre></details>
    </section>
    <section class="editor">
      <div class="meta"><span id="itemId"></span><span id="itemStatus"></span></div>
      <fieldset><legend>Outcome</legend><div class="segments" id="statusSegments"></div></fieldset>
      <fieldset id="eligibleFields">
        <legend>Source candidate</legend>
        <div class="option-list" id="candidateOptions"></div>
      </fieldset>
      <div class="field" id="canonicalField"><label for="canonical">Canonical source event</label><textarea id="canonical"></textarea></div>
      <div class="grid" id="realizationFields">
        <div class="field"><label for="acceptable">Acceptable Chinese realizations</label><textarea id="acceptable"></textarea></div>
        <div class="field"><label for="forbidden">Forbidden Chinese realizations</label><textarea id="forbidden"></textarea></div>
      </div>
      <div class="grid">
        <div class="field"><label for="alignment">Target alignment</label><select id="alignment"></select></div>
        <div class="field"><label for="slideStatus">Slide evidence</label><select id="slideStatus"></select></div>
      </div>
      <div class="field"><label for="note">Annotation note</label><textarea id="note"></textarea></div>
      <div class="actions"><button class="save" id="save">Save and continue</button><div class="toast" id="toast"></div></div>
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

function values(root) {
  return root.value.split("\n").map((value) => value.trim()).filter(Boolean);
}

function renderSegments(root, name, options, selected) {
  root.innerHTML = "";
  options.forEach((value) => {
    const label = document.createElement("label");
    label.className = "segment";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = value;
    input.checked = value === selected;
    input.onchange = renderMode;
    const span = document.createElement("span");
    span.textContent = value.replaceAll("_", " ");
    label.append(input, span);
    root.appendChild(label);
  });
}

function selectedStatus() {
  const selected = document.querySelector('input[name="status"]:checked');
  return selected ? selected.value : "eligible";
}

function renderMode() {
  const eligible = selectedStatus() === "eligible";
  $("#eligibleFields").classList.toggle("hidden", !eligible);
  $("#canonicalField").classList.toggle("hidden", !eligible);
  $("#realizationFields").classList.toggle("hidden", !eligible);
}

function renderOptions(options, selected) {
  const root = $("#candidateOptions");
  root.innerHTML = "";
  options.forEach((option) => {
    const label = document.createElement("label");
    label.className = "option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "candidate";
    input.value = option.option_id;
    input.checked = option.option_id === selected;
    const body = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = option.source_candidate_en;
    const small = document.createElement("small");
    small.textContent = `${option.candidate_kind} · ${option.token_count} tokens`;
    body.append(strong, small);
    const lead = document.createElement("span");
    lead.className = "lead";
    lead.textContent = `+${Number(option.lead_lower_bound_sec).toFixed(1)}s`;
    label.append(input, body, lead);
    root.appendChild(label);
  });
}

function renderSelect(root, options, selected) {
  root.innerHTML = '<option value="">Select</option>' + options.map(
    (value) => `<option value="${value}">${value.replaceAll("_", " ")}</option>`
  ).join("");
  root.value = selected || "";
}

async function load() {
  item = await api(`/api/item?index=${currentIndex}`);
  $("#progress").textContent = `${item.completed_count}/${item.total_count}`;
  $("#itemId").textContent = item.item_id;
  $("#itemStatus").textContent = item.annotation_status.replaceAll("_", " ");
  $("#itemStatus").className = item.annotation_status === "pending" ? "" : "done";
  $("#prev").disabled = currentIndex === 0;
  $("#next").disabled = currentIndex + 1 === item.total_count;
  $("#slide").src = `/api/media?index=${currentIndex}&v=${encodeURIComponent(item.item_id)}`;
  $("#source").textContent = item.source_reference_en;
  $("#target").textContent = item.target_reference_zh;
  $("#r0").textContent = item.current_slide_r0_text;
  $("#r1").textContent = item.current_slide_r1_text;
  const status = item.annotation_status === "pending" ? "eligible" : item.annotation_status;
  renderSegments($("#statusSegments"), "status", item.allowed_statuses, status);
  renderOptions(item.candidate_options, item.selected_option_id);
  $("#canonical").value = item.canonical_source_event_en || "";
  $("#acceptable").value = (item.acceptable_target_realizations_zh || []).join("\n");
  $("#forbidden").value = (item.forbidden_target_realizations_zh || []).join("\n");
  renderSelect($("#alignment"), item.allowed_target_alignments, item.target_reference_alignment);
  renderSelect($("#slideStatus"), item.allowed_slide_evidence_statuses, item.slide_evidence_status);
  $("#note").value = item.annotation_note || "";
  $("#toast").textContent = "";
  renderMode();
}

function setIndex(index) {
  currentIndex = Math.max(0, Math.min(item.total_count - 1, index));
  history.replaceState(null, "", `?index=${currentIndex}`);
  load().catch(showError);
}

function showError(error) {
  $("#toast").textContent = error.message;
}

$("#prev").onclick = () => setIndex(currentIndex - 1);
$("#next").onclick = () => setIndex(currentIndex + 1);
$("#save").onclick = async () => {
  const candidate = document.querySelector('input[name="candidate"]:checked');
  try {
    const result = await api("/api/save", {
      method: "POST",
      body: JSON.stringify({
        index: currentIndex,
        item_id: item.item_id,
        annotation_status: selectedStatus(),
        selected_option_id: candidate ? candidate.value : null,
        canonical_source_event_en: $("#canonical").value,
        acceptable_target_realizations_zh: values($("#acceptable")),
        forbidden_target_realizations_zh: values($("#forbidden")),
        target_reference_alignment: $("#alignment").value || null,
        slide_evidence_status: $("#slideStatus").value || null,
        annotation_note: $("#note").value,
      }),
    });
    $("#progress").textContent = `${result.item.completed_count}/${result.item.total_count}`;
    setIndex(result.next_index);
  } catch (error) {
    showError(error);
  }
};

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") $("#save").click();
  if (event.altKey && event.key === "ArrowLeft") setIndex(currentIndex - 1);
  if (event.altKey && event.key === "ArrowRight") setIndex(currentIndex + 1);
});

load().catch(showError);
"""


def make_handler(session: TargetEventAuthoringSession):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload: dict, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(payload, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def item_index(self) -> int:
            query = parse_qs(urlparse(self.path).query)
            values = query.get("index")
            if values is None or len(values) != 1:
                raise ValueError("MCIF author request requires one item index")
            return int(values[0])

        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                elif path == "/app.js":
                    self.send_bytes(APP_JS.encode(), "text/javascript; charset=utf-8")
                elif path == "/api/item":
                    self.send_json(session.state(self.item_index()))
                elif path == "/api/media":
                    self.send_bytes(session.media_path(self.item_index()).read_bytes(), "image/png")
                else:
                    self.send_json({"error": "Not found"}, 404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

        def do_POST(self) -> None:
            try:
                if urlparse(self.path).path != "/api/save":
                    self.send_json({"error": "Not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("MCIF author payload size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("MCIF author payload must be an object")
                item = session.save(payload)
                self.send_json(
                    {
                        "item": item,
                        "next_index": session.next_pending(payload["index"]),
                    }
                )
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sheet", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--working-sheet", type=Path, required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-items", type=int, default=355)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43871)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("MCIF target-event authoring server must bind localhost")
    session = TargetEventAuthoringSession(
        input_sheet=args.input_sheet,
        expected_input_sha256=args.expected_input_sha256,
        workspace_root=args.workspace_root,
        working_sheet=args.working_sheet,
        annotator_id=args.annotator_id,
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        expected_items=args.expected_items,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"MCIF target-event authoring: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
