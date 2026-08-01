#!/usr/bin/env python3
"""Serve causally gated ACL60/60 audio-only annotation tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse
import wave

from scripts.acl6060_source_event_annotation_v2 import (
    canonical_hash,
    expected_prefix_grid,
    index_unique,
    load_jsonl,
    question_only_lock_payload,
    sha256_file,
    verify_stage_media,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def event_hash(event: dict) -> str:
    payload = {key: value for key, value in event.items() if key != "event_sha256"}
    return canonical_hash(payload)


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = load_jsonl(path)
    previous = None
    for index, event in enumerate(events):
        if event.get("event_index") != index:
            raise ValueError(f"Non-contiguous event index: {index}")
        if event.get("previous_event_sha256") != previous:
            raise ValueError(f"Broken event chain: {index}")
        if event.get("event_sha256") != event_hash(event):
            raise ValueError(f"Event hash mismatch: {index}")
        previous = event["event_sha256"]
    return events


def append_event(path: Path, events: list[dict], payload: dict) -> dict:
    event = {
        **payload,
        "event_index": len(events),
        "server_time_utc": utc_now(),
        "previous_event_sha256": events[-1]["event_sha256"] if events else None,
    }
    event["event_sha256"] = event_hash(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    events.append(event)
    return event


def clipped_wav_bytes(path: Path, end_sec: float) -> bytes:
    output = io.BytesIO()
    with wave.open(str(path), "rb") as source:
        frame_count = min(source.getnframes(), round(end_sec * source.getframerate()))
        params = source.getparams()
        frames = source.readframes(frame_count)
    with wave.open(output, "wb") as destination:
        destination.setparams(params)
        destination.writeframes(frames)
    return output.getvalue()


class SequentialAudioSession:
    def __init__(
        self,
        tasks: list[dict],
        audio_root: Path,
        log_path: Path,
        output_path: Path,
    ) -> None:
        self.tasks = tasks
        self.task_index = index_unique(tasks)
        self.audio_root = audio_root
        self.log_path = log_path
        self.output_path = output_path
        self.lock = threading.Lock()
        validator_ids = {row["validator_id"] for row in tasks}
        if len(validator_ids) != 1:
            raise ValueError("Task sheet must contain exactly one validator id")
        self.validator_id = next(iter(validator_ids))
        verify_stage_media(tasks, audio_root, "audio_path", "audio_sha256")
        self.task_sheet_sha256 = canonical_hash({"tasks": tasks})
        self.events = read_events(log_path)
        if not self.events:
            append_event(
                self.log_path,
                self.events,
                {
                    "event_type": "session_started",
                    "validator_id": self.validator_id,
                    "task_sheet_sha256": self.task_sheet_sha256,
                },
            )
        first = self.events[0]
        if (
            first.get("event_type") != "session_started"
            or first.get("validator_id") != self.validator_id
            or first.get("task_sheet_sha256") != self.task_sheet_sha256
        ):
            raise ValueError("Interaction log belongs to another task sheet")
        self.recover_completed_items()
        if self.current_task() is None:
            self.export_complete_annotations()

    def recover_completed_items(self) -> None:
        for task in self.tasks:
            events = self.packet_events(task["packet_id"])
            if any(event["event_type"] == "item_completed" for event in events):
                continue
            question_count = sum(
                event["event_type"] == "question_only_submitted" for event in events
            )
            prefix_count = sum(
                event["event_type"] == "prefix_submitted" for event in events
            )
            if question_count == 1 and prefix_count == len(expected_prefix_grid(task)):
                append_event(
                    self.log_path,
                    self.events,
                    {
                        "event_type": "item_completed",
                        "packet_id": task["packet_id"],
                        "recovered_after_restart": True,
                    },
                )

    def packet_events(self, packet_id: str) -> list[dict]:
        return [event for event in self.events if event.get("packet_id") == packet_id]

    def current_task(self) -> dict | None:
        completed = {
            event["packet_id"]
            for event in self.events
            if event["event_type"] == "item_completed"
        }
        return next((row for row in self.tasks if row["packet_id"] not in completed), None)

    def state(self) -> dict:
        task = self.current_task()
        if task is None:
            return {
                "complete": True,
                "validator_id": self.validator_id,
                "completed_count": len(self.tasks),
                "total_count": len(self.tasks),
            }
        events = self.packet_events(task["packet_id"])
        question = next(
            (event for event in events if event["event_type"] == "question_only_submitted"),
            None,
        )
        prefix_events = [
            event for event in events if event["event_type"] == "prefix_submitted"
        ]
        grid = expected_prefix_grid(task)
        stage = "question_only" if question is None else "prefix"
        next_step = len(prefix_events) if question is not None else None
        return {
            "complete": False,
            "validator_id": self.validator_id,
            "completed_count": sum(
                event["event_type"] == "item_completed" for event in self.events
            ),
            "total_count": len(self.tasks),
            "packet_id": task["packet_id"],
            "source_question": task["source_question"],
            "source_options": task["source_options"],
            "stage": stage,
            "next_step_index": next_step,
            "prefix_step_count": len(grid),
        }

    def submit_question(self, payload: dict) -> None:
        with self.lock:
            state = self.state()
            if state.get("complete") or state["stage"] != "question_only":
                raise ValueError("Question-only stage is not active")
            if payload.get("packet_id") != state["packet_id"]:
                raise ValueError("Wrong current packet")
            status = payload.get("status")
            option_id = payload.get("option_id")
            option_ids = {option["option_id"] for option in state["source_options"]}
            if status not in {"not_answerable", "answerable", "ambiguous"}:
                raise ValueError("Invalid question-only status")
            if status == "answerable" and option_id not in option_ids:
                raise ValueError("Answerable question requires a valid option")
            if status != "answerable" and option_id is not None:
                raise ValueError("Non-answerable question cannot carry an option")
            append_event(
                self.log_path,
                self.events,
                {
                    "event_type": "question_only_submitted",
                    "packet_id": state["packet_id"],
                    "status": status,
                    "option_id": option_id,
                },
            )

    def submit_prefix(self, payload: dict) -> None:
        with self.lock:
            state = self.state()
            if state.get("complete") or state["stage"] != "prefix":
                raise ValueError("Prefix stage is not active")
            if payload.get("packet_id") != state["packet_id"]:
                raise ValueError("Wrong current packet")
            if payload.get("step_index") != state["next_step_index"]:
                raise ValueError("Prefix response is out of order")
            releases = [
                event
                for event in self.packet_events(state["packet_id"])
                if event["event_type"] == "prefix_released"
                and event["step_index"] == state["next_step_index"]
            ]
            if not releases:
                raise ValueError("Current prefix has not been released")
            task = self.task_index[state["packet_id"]]
            prefix_end_sec = expected_prefix_grid(task)[state["next_step_index"]]
            status = payload.get("status")
            option_id = payload.get("option_id")
            option_ids = {option["option_id"] for option in state["source_options"]}
            if status not in {"insufficient", "uncertain", "option"}:
                raise ValueError("Invalid prefix status")
            if status == "option" and option_id not in option_ids:
                raise ValueError("Option response requires a valid option")
            if status != "option" and option_id is not None:
                raise ValueError("Non-option response cannot carry an option")
            append_event(
                self.log_path,
                self.events,
                {
                    "event_type": "prefix_submitted",
                    "packet_id": state["packet_id"],
                    "step_index": state["next_step_index"],
                    "prefix_end_sec": prefix_end_sec,
                    "status": status,
                    "option_id": option_id,
                },
            )
            if state["next_step_index"] + 1 == state["prefix_step_count"]:
                append_event(
                    self.log_path,
                    self.events,
                    {
                        "event_type": "item_completed",
                        "packet_id": state["packet_id"],
                    },
                )
                if self.current_task() is None:
                    self.export_complete_annotations()

    def current_audio(self, packet_id: str) -> bytes:
        with self.lock:
            state = self.state()
            if state.get("complete") or state["stage"] != "prefix":
                raise ValueError("Audio is locked until question-only submission")
            if packet_id != state["packet_id"]:
                raise ValueError("Wrong current packet")
            task = self.task_index[packet_id]
            prefix_end_sec = expected_prefix_grid(task)[state["next_step_index"]]
            append_event(
                self.log_path,
                self.events,
                {
                    "event_type": "prefix_released",
                    "packet_id": packet_id,
                    "step_index": state["next_step_index"],
                    "prefix_end_sec": prefix_end_sec,
                },
            )
            return clipped_wav_bytes(
                self.audio_root / task["audio_path"],
                float(prefix_end_sec),
            )

    def export_complete_annotations(self) -> list[dict]:
        completed = {
            event["packet_id"]: event
            for event in self.events
            if event["event_type"] == "item_completed"
        }
        if set(completed) != set(self.task_index):
            raise ValueError("Cannot export an incomplete session")
        output = []
        interaction_log_sha256 = sha256_file(self.log_path)
        for task in self.tasks:
            packet_events = self.packet_events(task["packet_id"])
            question = next(
                event
                for event in packet_events
                if event["event_type"] == "question_only_submitted"
            )
            prefixes = [
                event
                for event in packet_events
                if event["event_type"] == "prefix_submitted"
            ]
            row = {
                **task,
                "question_only_status": question["status"],
                "question_only_answer_option_id": question["option_id"],
                "question_only_submitted_at_utc": question["server_time_utc"],
                "audio_annotation_status": "complete",
                "prefix_judgments": [
                    {
                        "step_index": event["step_index"],
                        "prefix_end_sec": event["prefix_end_sec"],
                        "status": event["status"],
                        "option_id": event["option_id"],
                    }
                    for event in prefixes
                ],
                "audio_submitted_at_utc": completed[task["packet_id"]][
                    "server_time_utc"
                ],
                "interaction_log_tail_sha256": completed[task["packet_id"]][
                    "event_sha256"
                ],
                "interaction_log_sha256": interaction_log_sha256,
                "sequential_delivery_backend": "acl6060_audio_gate_v1",
            }
            row["question_only_lock_sha256"] = canonical_hash(
                question_only_lock_payload(row)
            )
            output.append(row)
        write_jsonl(self.output_path, output)
        return output


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ACL Audio Annotation</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #f4f5f7; color: #17191c; }
    main { max-width: 760px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; border-bottom: 1px solid #c9cdd2; padding-bottom: 12px; }
    .panel { padding: 20px 0; }
    button { min-height: 40px; margin: 6px 8px 6px 0; padding: 8px 12px; border: 1px solid #8b929a; background: white; border-radius: 6px; }
    button.primary { background: #1769aa; color: white; border-color: #1769aa; }
    audio { width: 100%; margin: 16px 0; }
    .options { display: grid; gap: 8px; }
    .option { text-align: left; }
    small { color: #59616a; }
  </style>
</head>
<body>
  <main>
    <header><strong>ACL Audio Annotation</strong><span id="progress"></span></header>
    <section class="panel">
      <h2 id="question"></h2>
      <small id="prefix"></small>
      <div id="content"></div>
    </section>
  </main>
  <script src="/app.js"></script>
</body>
</html>"""


APP_JS = r"""
let state = null;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error((await response.json()).error);
  return response.json();
}

async function load() {
  state = await api("/api/state");
  document.querySelector("#progress").textContent =
    `${state.completed_count}/${state.total_count}`;
  const question = document.querySelector("#question");
  const content = document.querySelector("#content");
  const prefix = document.querySelector("#prefix");
  content.innerHTML = "";
  prefix.textContent = "";
  if (state.complete) {
    question.textContent = "Complete";
    return;
  }
  question.textContent = state.source_question;
  if (state.stage === "question_only") {
    renderQuestion(content);
  } else {
    prefix.textContent = `Prefix ${state.next_step_index + 1}/${state.prefix_step_count}`;
    renderPrefix(content);
  }
}

function optionButtons(parent, callback) {
  const box = document.createElement("div");
  box.className = "options";
  state.source_options.forEach((option) => {
    const button = document.createElement("button");
    button.className = "option";
    button.textContent = option.text;
    button.onclick = () => callback(option.option_id);
    box.appendChild(button);
  });
  parent.appendChild(box);
}

function post(path, data) {
  return api(path, { method: "POST", body: JSON.stringify(data) })
    .then(load)
    .catch((error) => alert(error.message));
}

function renderQuestion(content) {
  const notAnswerable = document.createElement("button");
  notAnswerable.textContent = "Not answerable";
  notAnswerable.className = "primary";
  notAnswerable.onclick = () => post("/api/question", {
    packet_id: state.packet_id,
    status: "not_answerable",
    option_id: null,
  });
  content.appendChild(notAnswerable);

  const ambiguous = document.createElement("button");
  ambiguous.textContent = "Ambiguous";
  ambiguous.onclick = () => post("/api/question", {
    packet_id: state.packet_id,
    status: "ambiguous",
    option_id: null,
  });
  content.appendChild(ambiguous);
  optionButtons(content, (optionId) => post("/api/question", {
    packet_id: state.packet_id,
    status: "answerable",
    option_id: optionId,
  }));
}

function renderPrefix(content) {
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src =
    `/api/audio?packet_id=${encodeURIComponent(state.packet_id)}` +
    `&v=${state.next_step_index}`;
  content.appendChild(audio);

  for (const [label, status] of [
    ["Insufficient", "insufficient"],
    ["Uncertain", "uncertain"],
  ]) {
    const button = document.createElement("button");
    button.textContent = label;
    if (status === "insufficient") button.className = "primary";
    button.onclick = () => post("/api/prefix", {
      packet_id: state.packet_id,
      step_index: state.next_step_index,
      status,
      option_id: null,
    });
    content.appendChild(button);
  }
  optionButtons(content, (optionId) => post("/api/prefix", {
    packet_id: state.packet_id,
    step_index: state.next_step_index,
    status: "option",
    option_id: optionId,
  }));
}

load().catch((error) => alert(error.message));
"""


def make_handler(session: SequentialAudioSession):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: dict, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(value).encode("utf-8"), "application/json", status
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                elif parsed.path == "/app.js":
                    self.send_bytes(APP_JS.encode(), "text/javascript; charset=utf-8")
                elif parsed.path == "/api/state":
                    self.send_json(session.state())
                elif parsed.path == "/api/audio":
                    packet_id = parse_qs(parsed.query).get("packet_id", [None])[0]
                    self.send_bytes(session.current_audio(packet_id), "audio/wav")
                else:
                    self.send_json({"error": "Not found"}, 404)
            except (ValueError, KeyError) as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("Request body too large")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/api/question":
                    session.submit_question(payload)
                elif self.path == "/api/prefix":
                    session.submit_prefix(payload)
                else:
                    self.send_json({"error": "Not found"}, 404)
                    return
                self.send_json({"ok": True})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-sheet", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    tasks = load_jsonl(args.task_sheet)
    if not tasks:
        raise ValueError("Task sheet is empty")
    session = SequentialAudioSession(tasks, args.audio_root, args.event_log, args.output)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
