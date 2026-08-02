#!/usr/bin/env python3
"""Serve one leak-resistant MCIF reliability-v2 annotation role and stage."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image
from scripts.build_mcif_visual_token_controls import file_sha256, load_jsonl
from scripts.mcif_beyond_ocr_reliability import (
    append_event_log,
    blank_annotation,
    input_contract,
    load_config,
    load_hmac_key,
    load_identity_registry,
    load_run_contract,
    registered_annotator_id,
    validate_contract_stage_item_count,
    validate_event_head_ledger,
    validate_event_log,
    validate_run_contract,
)

COOKIE_NAME = "mcif_reliability_v2_session"
MAX_PAYLOAD_BYTES = 1024 * 1024
POST_KEYS = {
    "item_id",
    "expected_event_index",
    "annotation_status",
    "annotation",
}
IMAGE_CONTENT_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
R1_BLOCK_FIELDS = (
    "content_kind",
    "label",
    "content",
    "text",
    "reading_order",
)
ORIGIN_FIELDS = (
    "descriptor_field",
    "descriptor_index",
    "descriptor_text",
    "block_id",
    "content",
    "content_kind",
    "label",
)
SCORING_TEXT_FIELDS = (
    "canonical_source_event_en",
    "acceptable_target_realizations_zh",
    "forbidden_target_realizations_zh",
)


@dataclass(frozen=True)
class BoundMedia:
    path: Path
    sha256: str
    content_type: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick(mapping: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {field: mapping[field] for field in fields if field in mapping}


def _project_list(values: Any, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [_pick(value, fields) for value in values if isinstance(value, dict)]


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"MCIF reliability-v2 {label} must be a regular file")
    return path.resolve(strict=True)


def _secret_file(path: Path, label: str) -> Path:
    resolved = _regular_file(path, label)
    if stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        raise ValueError(f"MCIF reliability-v2 {label} permissions must be 0600")
    return resolved


def _workspace_file(root: Path, raw_relative: Any) -> Path:
    if not isinstance(raw_relative, str) or not raw_relative:
        raise ValueError("MCIF reliability-v2 media path is absent")
    relative = PurePosixPath(raw_relative)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != raw_relative
    ):
        raise ValueError(
            "MCIF reliability-v2 media path must be canonical and relative"
        )
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("MCIF reliability-v2 media path cannot traverse a symlink")
    resolved = current.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("MCIF reliability-v2 media path escapes the workspace root")
    return resolved


def _bind_media(root: Path, descriptor: Any) -> BoundMedia:
    if not isinstance(descriptor, dict):
        raise ValueError("MCIF reliability-v2 image descriptor is absent")
    expected_sha256 = descriptor.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("MCIF reliability-v2 image hash is invalid")
    path = _workspace_file(root, descriptor.get("path"))
    if file_sha256(path) != expected_sha256:
        raise ValueError("MCIF reliability-v2 image hash differs from input contract")
    with Image.open(path) as image:
        image_format = image.format
        size = image.size
        image.verify()
    if image_format not in IMAGE_CONTENT_TYPES:
        raise ValueError("MCIF reliability-v2 media is not a supported image")
    expected_size = (descriptor.get("width"), descriptor.get("height"))
    if all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in expected_size
    ):
        if size != expected_size:
            raise ValueError(
                "MCIF reliability-v2 image dimensions differ from input contract"
            )
    return BoundMedia(path, expected_sha256, IMAGE_CONTENT_TYPES[image_format])


class ReliabilityAnnotationSession:
    def __init__(
        self,
        *,
        input_sheet: Path,
        expected_input_sha256: str,
        workspace_root: Path,
        event_log: Path,
        head_ledger: Path,
        annotator_id: str,
        config_path: Path,
        expected_config_sha256: str,
        hmac_key_path: Path,
        expected_items: int,
        expected_role: str | None = None,
        expected_stage: str | None = None,
        identity_registry_path: Path,
        expected_identity_registry_sha256: str,
        run_contract_path: Path,
        expected_run_contract_file_sha256: str,
        access_token_path: Path,
    ) -> None:
        if workspace_root.is_symlink() or not workspace_root.is_dir():
            raise ValueError(
                "MCIF reliability-v2 workspace root must be a real directory"
            )
        self.workspace_root = workspace_root.resolve(strict=True)
        resolved_input = _regular_file(input_sheet, "input sheet")
        if not resolved_input.is_relative_to(self.workspace_root):
            raise ValueError("MCIF reliability-v2 input sheet is outside its workspace")
        if file_sha256(resolved_input) != expected_input_sha256:
            raise ValueError("MCIF reliability-v2 input sheet hash differs")
        _regular_file(config_path, "config")
        _secret_file(hmac_key_path, "HMAC key")
        self.config = load_config(config_path, expected_config_sha256)
        self.key = load_hmac_key(hmac_key_path)
        self.input_rows = load_jsonl(resolved_input)
        contracts = {input_contract(row) for row in self.input_rows}
        if len(contracts) != 1:
            raise ValueError(
                "MCIF reliability-v2 server requires one role/stage contract"
            )
        self.role, self.stage = next(iter(contracts))
        if expected_role is not None and self.role != expected_role:
            raise ValueError(
                "MCIF reliability-v2 server role differs from launch contract"
            )
        if expected_stage is not None and self.stage != expected_stage:
            raise ValueError(
                "MCIF reliability-v2 server stage differs from launch contract"
            )
        self.identity_registry = load_identity_registry(
            identity_registry_path, expected_identity_registry_sha256, self.config
        )
        self.run_contract = load_run_contract(
            run_contract_path,
            expected_run_contract_file_sha256,
            key=self.key,
            config=self.config,
            identity_registry=self.identity_registry,
        )
        self.run_contract_sha256 = validate_run_contract(
            self.run_contract,
            key=self.key,
            config=self.config,
            identity_registry=self.identity_registry,
        )
        validate_contract_stage_item_count(
            self.run_contract, expected_items=expected_items, role=self.role
        )
        resolved_access_token = _secret_file(access_token_path, "access token")
        self.access_token_path = resolved_access_token
        access_token = resolved_access_token.read_bytes()
        if (
            len(access_token) < 32
            or hashlib.sha256(access_token).hexdigest()
            != self.run_contract["role_access_token_sha256"][self.role]
        ):
            raise ValueError(
                "MCIF reliability-v2 access token differs from run contract"
            )
        try:
            self.access_token = access_token.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("MCIF reliability-v2 access token must be ASCII") from exc
        self.event_log = _regular_file(event_log, "event log")
        self.head_ledger = _secret_file(head_ledger, "event-head ledger")
        self.annotator_id = annotator_id
        if annotator_id != registered_annotator_id(
            self.identity_registry, self.config, role=self.role
        ):
            raise ValueError(
                "MCIF reliability-v2 server annotator differs from identity registry"
            )
        self.expected_items = expected_items
        self.events = load_jsonl(self.event_log)
        self.head_checkpoints = load_jsonl(self.head_ledger)
        self._lock = threading.RLock()
        self._grouped = validate_event_log(
            self.events,
            self.input_rows,
            annotator_id=annotator_id,
            expected_items=expected_items,
            key=self.key,
            config=self.config,
            run_contract_sha256=self.run_contract_sha256,
        )
        validate_event_head_ledger(
            self.head_checkpoints,
            self.events,
            self.input_rows,
            annotator_id=annotator_id,
            expected_items=expected_items,
            key=self.key,
            config=self.config,
            run_contract_sha256=self.run_contract_sha256,
        )
        self._input_by_id = {row["item_id"]: row for row in self.input_rows}
        self._index_by_id = {
            row["item_id"]: index for index, row in enumerate(self.input_rows)
        }
        self._media_by_id: dict[str, BoundMedia] = {}
        for row in self.input_rows:
            descriptor = self._media_descriptor(row)
            if descriptor is not None:
                self._media_by_id[row["item_id"]] = _bind_media(
                    self.workspace_root, descriptor
                )
        self.session_token = secrets.token_urlsafe(32)

    def _media_descriptor(self, row: dict[str, Any]) -> dict[str, Any] | None:
        role, stage = input_contract(row)
        if role in {"visual_a", "visual_b"} and stage in {"pixels", "descriptor"}:
            return row.get("current_slide")
        if role == "visual_adjudicator":
            evidence = row.get("released_evidence")
            if isinstance(evidence, dict):
                return evidence.get("current_slide")
        return None

    def _latest(self, item_id: str) -> dict[str, Any]:
        return self._grouped[item_id][-1]

    def progress(self) -> dict[str, int]:
        completed = sum(
            self._latest(row["item_id"])["annotation_status"] == "completed"
            for row in self.input_rows
        )
        return {"completed": completed, "total": len(self.input_rows)}

    def _project_visual(self, row: dict[str, Any], output: dict[str, Any]) -> None:
        output["r0_text"] = row["r0_text"]
        if self.stage != "r0":
            locked = row.get("locked_judgments")
            if isinstance(locked, dict):
                output["locked_judgments"] = {
                    key: locked[key]
                    for key in (
                        "r0_support",
                        "r1_support",
                        "pixel_support",
                        "descriptor_fidelity",
                    )
                    if key in locked
                }
        if self.stage in {"r1", "pixels", "descriptor"}:
            output["r1_blocks"] = _project_list(row["r1_blocks"], R1_BLOCK_FIELDS)
        if self.stage in {"pixels", "descriptor"}:
            output["image_available"] = row["item_id"] in self._media_by_id
        if self.stage == "descriptor":
            output["proposed_evidence_origins"] = _project_list(
                row["proposed_evidence_origins"], ORIGIN_FIELDS
            )

    def _project_target(self, row: dict[str, Any], output: dict[str, Any]) -> None:
        output["source_reference_en"] = row["source_reference_en"]
        output["target_reference_zh"] = row["target_reference_zh"]
        if self.stage == "author_text_review":
            output["locked_candidate_eligibility"] = row["locked_candidate_eligibility"]
            output["locked_target_reference_alignment"] = row[
                "locked_target_reference_alignment"
            ]
            output["author_candidate_eligibility"] = row["author_candidate_eligibility"]
            output["author_target_reference_alignment"] = row[
                "author_target_reference_alignment"
            ]
            output["author_scoring_text"] = {
                "canonical_source_event_en": row["author_canonical_source_event_en"],
                "acceptable_target_realizations_zh": row[
                    "author_acceptable_target_realizations_zh"
                ],
                "forbidden_target_realizations_zh": row[
                    "author_forbidden_target_realizations_zh"
                ],
            }

    def _project_visual_adjudication(
        self, row: dict[str, Any], output: dict[str, Any]
    ) -> None:
        evidence = row["released_evidence"]
        output["primitive_field"] = row["primitive_field"]
        output["released_evidence"] = {
            "r0_text": evidence.get("r0_text", ""),
        }
        if "r1_blocks" in evidence:
            output["released_evidence"]["r1_blocks"] = _project_list(
                evidence["r1_blocks"], R1_BLOCK_FIELDS
            )
        if "proposed_evidence_origins" in evidence:
            output["released_evidence"]["proposed_evidence_origins"] = _project_list(
                evidence["proposed_evidence_origins"], ORIGIN_FIELDS
            )
        if row["item_id"] in self._media_by_id:
            output["released_evidence"]["image_available"] = True
        raw_fields = ("judgment", "reason_codes", "note")
        output["visual_a_raw"] = _pick(row["visual_a_raw"], raw_fields)
        output["visual_b_raw"] = _pick(row["visual_b_raw"], raw_fields)

    def _project_target_adjudication(
        self, row: dict[str, Any], output: dict[str, Any]
    ) -> None:
        output["released_source"] = _pick(
            row["released_source"], ("source_reference_en", "target_reference_zh")
        )
        target_raw = row["target_raw"]
        output["target_raw"] = {
            "candidate_eligibility": _pick(
                target_raw.get("candidate_eligibility"), ("author", "validator")
            ),
            "target_reference_alignment": _pick(
                target_raw.get("target_reference_alignment"), ("author", "validator")
            ),
            "stage2_review_decision": target_raw.get("stage2_review_decision"),
            "author_reason_codes": target_raw.get("author_reason_codes", []),
            "validator_stage1_reason_codes": target_raw.get(
                "validator_stage1_reason_codes", []
            ),
            "validator_stage2_reason_codes": target_raw.get(
                "validator_stage2_reason_codes", []
            ),
        }
        output["author_scoring_text"] = _pick(
            row["author_scoring_text"], SCORING_TEXT_FIELDS
        )
        output["validator_edits"] = _pick(row["validator_edits"], SCORING_TEXT_FIELDS)

    def _public_item(self, row: dict[str, Any]) -> dict[str, Any]:
        latest = self._latest(row["item_id"])
        output = {
            "item_id": row["item_id"],
            "candidate_source_en": row["candidate_source_en"],
            "candidate_kind": row["candidate_kind"],
            "candidate_token_count": row["candidate_token_count"],
            "annotation_status": latest["annotation_status"],
            "event_index": latest["event_index"],
            "annotation": {
                key: latest[key] for key in blank_annotation(self.role, self.stage)
            },
        }
        if self.role in {"visual_a", "visual_b"}:
            self._project_visual(row, output)
        elif self.role in {"target_author", "target_validator"}:
            self._project_target(row, output)
        elif self.role == "visual_adjudicator":
            self._project_visual_adjudication(row, output)
        else:
            self._project_target_adjudication(row, output)
        return output

    def state(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("MCIF reliability-v2 item index must be an integer")
        with self._lock:
            if index < 0 or index >= len(self.input_rows):
                raise ValueError("MCIF reliability-v2 item index is outside the input")
            return {
                "role": self.role,
                "stage": self.stage,
                "index": index,
                "item": self._public_item(self.input_rows[index]),
                "progress": self.progress(),
            }

    def metadata(self) -> dict[str, Any]:
        visual_role = self.role in {"visual_a", "visual_b", "visual_adjudicator"}
        return {
            "role": self.role,
            "stage": self.stage,
            "reason_codes": list(
                self.config["visual" if visual_role else "target"]["reason_codes"]
            ),
        }

    def next_open(self, current: int) -> int:
        with self._lock:
            for offset in range(1, len(self.input_rows) + 1):
                index = (current + offset) % len(self.input_rows)
                if (
                    self._latest(self.input_rows[index]["item_id"])["annotation_status"]
                    != "completed"
                ):
                    return index
            return current

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != POST_KEYS:
            raise ValueError("MCIF reliability-v2 POST payload keys differ")
        item_id = payload["item_id"]
        expected_event_index = payload["expected_event_index"]
        annotation_status = payload["annotation_status"]
        annotation = payload["annotation"]
        if not isinstance(item_id, str) or item_id not in self._input_by_id:
            raise ValueError("MCIF reliability-v2 submitted item differs")
        if not isinstance(expected_event_index, int) or isinstance(
            expected_event_index, bool
        ):
            raise ValueError("MCIF reliability-v2 expected event index is invalid")
        if annotation_status not in {"draft", "completed"}:
            raise ValueError("MCIF reliability-v2 annotation status differs")
        if not isinstance(annotation, dict):
            raise ValueError("MCIF reliability-v2 annotation must be an object")
        with self._lock:
            replacement = append_event_log(
                self.event_log,
                self.head_ledger,
                self.input_rows,
                item_id=item_id,
                expected_event_index=expected_event_index,
                annotation_status=annotation_status,
                annotation=annotation,
                submitted_at_utc=_utc_now(),
                annotator_id=self.annotator_id,
                expected_items=self.expected_items,
                key=self.key,
                config=self.config,
                run_contract_sha256=self.run_contract_sha256,
            )
            grouped = validate_event_log(
                replacement,
                self.input_rows,
                annotator_id=self.annotator_id,
                expected_items=self.expected_items,
                key=self.key,
                config=self.config,
                run_contract_sha256=self.run_contract_sha256,
            )
            self.events = replacement
            self.head_checkpoints = load_jsonl(self.head_ledger)
            self._grouped = grouped
            index = self._index_by_id[item_id]
            return {
                "state": self.state(index),
                "next_index": self.next_open(index),
            }

    def media_bytes(self, item_id: str) -> tuple[bytes, str]:
        with self._lock:
            media = self._media_by_id.get(item_id)
            if media is None:
                raise FileNotFoundError(
                    "MCIF reliability-v2 item has no released image"
                )
            payload = media.path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != media.sha256:
                raise ValueError(
                    "MCIF reliability-v2 image hash differs from input contract"
                )
            return payload, media.content_type

    def cookie_matches(self, raw_cookie: str | None) -> bool:
        if not raw_cookie:
            return False
        parsed = cookies.SimpleCookie()
        try:
            parsed.load(raw_cookie)
        except cookies.CookieError:
            return False
        morsel = parsed.get(COOKIE_NAME)
        return morsel is not None and hmac.compare_digest(
            morsel.value, self.session_token
        )

    def access_token_matches(self, candidate: str) -> bool:
        return hmac.compare_digest(candidate, self.access_token)


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCIF Beyond-OCR Reliability v2</title><link rel="stylesheet" href="/app.css"></head>
<body><header><div><h1 id="title">MCIF Reliability v2</h1><span id="stage"></span></div><div id="progress">0 / 0</div><nav><button id="prev" title="Previous">&#8592;</button><button id="next" title="Next">&#8594;</button></nav></header>
<main><section class="candidate"><span>Candidate</span><strong id="candidate"></strong><small id="item-id"></small></section><section id="evidence" class="evidence"></section><section id="form" class="form"></section></main>
<footer><span id="message"></span><div><button id="draft">Save draft</button><button id="complete" class="primary">Complete</button></div></footer><script src="/app.js"></script></body></html>
"""


LOGIN_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCIF Reliability v2</title><link rel="stylesheet" href="/app.css"></head>
<body class="login"><main><form method="post" action="/login"><h1>MCIF Reliability v2</h1><label>Access token<input name="access_token" type="password" autocomplete="current-password" required></label><button class="primary" type="submit">Sign in</button></form></main></body></html>
"""


CSS = """
:root{--bg:#f3f5f7;--paper:#fff;--ink:#17212b;--muted:#617080;--line:#d5dce3;--accent:#08745e;--danger:#a13028}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}header{position:sticky;top:0;z-index:2;display:grid;grid-template-columns:1fr auto auto;align-items:center;gap:18px;padding:12px 20px;background:var(--paper);border-bottom:1px solid var(--line)}h1{margin:0;font-size:17px}header span,#progress,small,#message{color:var(--muted)}nav,footer div{display:flex;gap:8px}button{min-height:36px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:0 13px;font:inherit;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}.primary{border-color:var(--accent);background:var(--accent);color:#fff}main{max-width:1380px;margin:0 auto;padding-bottom:88px}.candidate{display:grid;grid-template-columns:auto 1fr auto;align-items:baseline;gap:14px;padding:18px 22px;background:#fff;border-bottom:1px solid var(--line)}.candidate strong{font-size:20px}.evidence{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));background:#fff;border-bottom:1px solid var(--line)}article{min-width:0;padding:18px 22px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}h2{margin:0 0 9px;color:var(--muted);font-size:12px;text-transform:uppercase}pre{margin:0;white-space:pre-wrap;word-break:break-word;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}.slide{grid-column:1/-1;background:#20262c;text-align:center}.slide img{display:block;max-width:100%;max-height:62vh;margin:auto}.form{padding:20px 22px}.control{margin-bottom:16px}.control>label{display:block;margin-bottom:7px;font-weight:600}.segments{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));max-width:620px}.segments label{position:relative}.segments input{position:absolute;opacity:0}.segments b{display:block;padding:9px;text-align:center;border:1px solid var(--line);border-right:0;background:#fff;font-weight:500;cursor:pointer}.segments label:last-child b{border-right:1px solid var(--line)}.segments input:checked+b{background:#dcefe9;border-color:var(--accent);color:#075241}.text-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.text-grid label{font-weight:600}textarea,input[type=text],input[type=password],select{display:block;width:100%;margin-top:6px;border:1px solid var(--line);background:#fff;padding:9px;font:inherit;border-radius:0}textarea{resize:vertical}.reasons{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.reasons label{border:1px solid var(--line);background:#fff;padding:7px 10px}footer{position:fixed;z-index:3;left:0;right:0;bottom:0;display:flex;justify-content:space-between;align-items:center;padding:12px 20px;background:#fff;border-top:1px solid var(--line)}.error{color:var(--danger)}.login main{display:grid;min-height:100vh;place-items:center;padding:20px}.login form{width:min(380px,100%);padding:24px;background:#fff;border:1px solid var(--line)}.login form h1{margin-bottom:20px}.login form label{display:block;font-weight:600}.login form button{width:100%;margin-top:16px}@media(max-width:760px){header{grid-template-columns:1fr auto;padding:10px 12px}#progress{grid-row:2}.candidate{grid-template-columns:1fr;padding:14px}.evidence,.text-grid{grid-template-columns:1fr}.slide{grid-column:auto}.slide img{max-height:44vh}.form{padding:16px 14px 96px}footer{padding:10px 12px}}
"""


APP_JS = r"""
let meta=null,state=null,current=0;
const $=id=>document.getElementById(id);const esc=value=>String(value??'').replace(/[&<>\"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[char]));
const lines=value=>String(value||'').split(/\n/).map(item=>item.trim()).filter(Boolean);
function article(title,value){return `<article><h2>${esc(title)}</h2><pre>${esc(typeof value==='string'?value:JSON.stringify(value,null,2))}</pre></article>`}
function radio(name,label,values,currentValue){return `<div class="control"><label>${esc(label)}</label><div class="segments">${values.map(value=>`<label><input type="radio" name="${esc(name)}" value="${esc(value)}" ${currentValue===value?'checked':''}><b>${esc(value)}</b></label>`).join('')}</div></div>`}
function text(name,label,value,multiline=false){return `<label>${esc(label)}${multiline?`<textarea name="${esc(name)}" rows="4">${esc(Array.isArray(value)?value.join('\n'):value||'')}</textarea>`:`<input type="text" name="${esc(name)}" value="${esc(value||'')}">`}</label>`}
function select(name,label,values,currentValue){return `<label>${esc(label)}<select name="${esc(name)}"><option value="">Select</option>${values.map(value=>`<option value="${esc(value)}" ${currentValue===value?'selected':''}>${esc(value)}</option>`).join('')}</select></label>`}
function evidence(item){let html='';if(item.r0_text!==undefined)html+=article('R0 flat OCR',item.r0_text);if(item.locked_judgments)html+=article('Locked prior judgments',item.locked_judgments);if(item.r1_blocks)html+=article('R1 structured text',item.r1_blocks);if(item.image_available||item.released_evidence?.image_available)html+=`<article class="slide"><img alt="Released slide evidence" src="/api/media?item_id=${encodeURIComponent(item.item_id)}"></article>`;if(item.proposed_evidence_origins)html+=article('Source-only descriptor',item.proposed_evidence_origins);if(item.source_reference_en!==undefined){html+=article('English source',item.source_reference_en);html+=article('Chinese reference',item.target_reference_zh)}if(item.locked_candidate_eligibility!==undefined)html+=article('Locked independent labels',{candidate_eligibility:item.locked_candidate_eligibility,target_reference_alignment:item.locked_target_reference_alignment});if(item.author_candidate_eligibility!==undefined)html+=article('Author labels',{candidate_eligibility:item.author_candidate_eligibility,target_reference_alignment:item.author_target_reference_alignment});if(item.author_scoring_text)html+=article('Author scoring text',item.author_scoring_text);if(item.primitive_field)html+=article('Primitive under adjudication',item.primitive_field);if(item.released_evidence){html+=article('Released evidence',item.released_evidence)}if(item.visual_a_raw)html+=article('Visual A raw',item.visual_a_raw);if(item.visual_b_raw)html+=article('Visual B raw',item.visual_b_raw);if(item.released_source)html+=article('Released source',item.released_source);if(item.target_raw)html+=article('Raw target judgments',item.target_raw);if(item.validator_edits)html+=article('Validator edits',item.validator_edits);return html}
function form(item){const a=item.annotation;let html='';if(meta.role==='visual_a'||meta.role==='visual_b'){const field={r0:'r0_support',r1:'r1_support',pixels:'pixel_support',descriptor:'descriptor_fidelity'}[meta.stage];html+=radio(field,field.replaceAll('_',' '),['yes','no','uncertain'],a[field])}else if(meta.role==='target_author'){html+=radio('candidate_eligibility','Candidate eligibility',['yes','no','uncertain'],a.candidate_eligibility);html+=`<div class="text-grid">${text('canonical_source_event_en','Canonical English event',a.canonical_source_event_en)}${select('target_reference_alignment','Target alignment',['explicit','paraphrased','omitted','unsupported','uncertain'],a.target_reference_alignment)}${text('acceptable_target_realizations_zh','Acceptable Chinese realizations',a.acceptable_target_realizations_zh,true)}${text('forbidden_target_realizations_zh','Forbidden Chinese realizations',a.forbidden_target_realizations_zh,true)}</div>`}else if(meta.role==='target_validator'&&meta.stage==='independent_alignment'){html+=radio('candidate_eligibility','Candidate eligibility',['yes','no','uncertain'],a.candidate_eligibility);html+=select('target_reference_alignment','Target alignment',['explicit','paraphrased','omitted','unsupported','uncertain'],a.target_reference_alignment)}else if(meta.role==='target_validator'){html+=radio('review_decision','Review decision',['accept','edit','reject'],a.review_decision);html+=`<div class="text-grid">${text('edited_canonical_source_event_en','Edited canonical event',a.edited_canonical_source_event_en)}${text('edited_acceptable_target_realizations_zh','Edited acceptable realizations',a.edited_acceptable_target_realizations_zh,true)}${text('edited_forbidden_target_realizations_zh','Edited forbidden realizations',a.edited_forbidden_target_realizations_zh,true)}</div>`}else if(meta.role==='visual_adjudicator'){html+=radio('adjudicated_judgment','Adjudicated judgment',['yes','no','unresolvable'],a.adjudicated_judgment)}else{html+=radio('adjudication_decision','Adjudication decision',['accept','edit','reject','unresolvable'],a.adjudication_decision);html+=`<div class="text-grid">${text('final_canonical_source_event_en','Final canonical event',a.final_canonical_source_event_en)}${text('final_acceptable_target_realizations_zh','Final acceptable realizations',a.final_acceptable_target_realizations_zh,true)}${text('final_forbidden_target_realizations_zh','Final forbidden realizations',a.final_forbidden_target_realizations_zh,true)}</div>`}html+=`<div class="reasons">${meta.reason_codes.map(reason=>`<label><input type="checkbox" name="reason_codes" value="${esc(reason)}" ${(a.reason_codes||[]).includes(reason)?'checked':''}>${esc(reason.replaceAll('_',' '))}</label>`).join('')}</div>${text('annotation_note','Annotation note',a.annotation_note,true)}`;return html}
function annotation(){const output=structuredClone(state.item.annotation);for(const key of Object.keys(output)){const radioInput=document.querySelector(`input[type=radio][name="${key}"]:checked`);const input=document.querySelector(`[name="${key}"]:not([type=radio])`);if(radioInput)output[key]=radioInput.value;else if(input){output[key]=Array.isArray(output[key])?lines(input.value):input.value}}output.reason_codes=[...document.querySelectorAll('input[name=reason_codes]:checked')].map(node=>node.value);return output}
function render(){const item=state.item;$('title').textContent=meta.role.replaceAll('_',' ');$('stage').textContent=meta.stage.replaceAll('_',' ');$('progress').textContent=`${state.progress.completed} / ${state.progress.total}`;$('candidate').textContent=item.candidate_source_en;$('item-id').textContent=`${item.item_id} · ${item.annotation_status}`;$('evidence').innerHTML=evidence(item);$('form').innerHTML=form(item);const done=item.annotation_status==='completed';$('draft').disabled=done;$('complete').disabled=done;$('message').textContent=done?'Completed and immutable':''}
async function jsonFetch(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw Error(data.error||`HTTP ${response.status}`);return data}
async function load(index){try{state=await jsonFetch(`/api/item?index=${index}`);current=state.index;render()}catch(error){$('message').className='error';$('message').textContent=error.message}}
async function save(status){try{const payload={item_id:state.item.item_id,expected_event_index:state.item.event_index,annotation_status:status,annotation:annotation()};const result=await jsonFetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});state=result.state;$('message').className='';if(status==='completed')await load(result.next_index);else render()}catch(error){$('message').className='error';$('message').textContent=error.message}}
async function start(){meta=await jsonFetch('/api/meta');$('prev').addEventListener('click',()=>load(Math.max(0,current-1)));$('next').addEventListener('click',()=>load(Math.min(state.progress.total-1,current+1)));$('draft').addEventListener('click',()=>save('draft'));$('complete').addEventListener('click',()=>save('completed'));await load(0)}start().catch(error=>{$('message').className='error';$('message').textContent=error.message});
"""


def make_handler(session: ReliabilityAnnotationSession):
    class Handler(BaseHTTPRequestHandler):
        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
            )

        def _send(
            self,
            payload: bytes,
            content_type: str,
            status: int = 200,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._send(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def _login_redirect(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self._security_headers()
            self.send_header(
                "Set-Cookie",
                f"{COOKIE_NAME}={session.session_token}; Path=/; HttpOnly; SameSite=Strict",
            )
            self.end_headers()

        def _authenticated(self) -> bool:
            return session.cookie_matches(self.headers.get("Cookie"))

        def _allowed_origin(self) -> bool:
            origin = self.headers.get("Origin")
            port = self.server.server_address[1]
            return origin in {
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
                f"http://[::1]:{port}",
            }

        def _index(self) -> int:
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            if set(query) != {"index"} or len(query["index"]) != 1:
                raise ValueError("MCIF reliability-v2 request requires one index")
            return int(query["index"][0])

        def _media_item_id(self) -> str:
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            if set(query) != {"item_id"} or len(query["item_id"]) != 1:
                raise ValueError(
                    "MCIF reliability-v2 media request requires one item id"
                )
            return query["item_id"][0]

        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    self._send(
                        (
                            HTML.encode("utf-8")
                            if self._authenticated()
                            else LOGIN_HTML.encode("utf-8")
                        ),
                        "text/html; charset=utf-8",
                    )
                    return
                if path == "/app.css":
                    self._send(CSS.encode("utf-8"), "text/css; charset=utf-8")
                    return
                if not self._authenticated():
                    self._json({"error": "Forbidden"}, 403)
                    return
                if path == "/app.js":
                    self._send(APP_JS.encode("utf-8"), "text/javascript; charset=utf-8")
                elif path == "/api/meta":
                    self._json(session.metadata())
                elif path == "/api/item":
                    self._json(session.state(self._index()))
                elif path == "/api/media":
                    payload, content_type = session.media_bytes(self._media_item_id())
                    self._send(payload, content_type)
                else:
                    self._json({"error": "Not found"}, 404)
            except FileNotFoundError as exc:
                self._json({"error": str(exc)}, 404)
            except Exception as exc:
                self._json({"error": str(exc)}, 400)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/login":
                if not self._allowed_origin():
                    self._json({"error": "Forbidden"}, 403)
                    return
                if (
                    self.headers.get("Content-Type", "").strip().lower()
                    != "application/x-www-form-urlencoded"
                ):
                    self._json({"error": "Content-Type must be form-urlencoded"}, 415)
                    return
                try:
                    raw_length = self.headers.get("Content-Length")
                    length = int(raw_length) if raw_length is not None else 0
                    if length <= 0 or length > 4096:
                        raise ValueError(
                            "MCIF reliability-v2 login payload size is invalid"
                        )
                    form = parse_qs(
                        self.rfile.read(length).decode("ascii"),
                        keep_blank_values=True,
                        strict_parsing=True,
                    )
                    if set(form) != {"access_token"} or len(form["access_token"]) != 1:
                        raise ValueError("MCIF reliability-v2 login payload differs")
                    if not session.access_token_matches(form["access_token"][0]):
                        self._json({"error": "Forbidden"}, 403)
                        return
                    self._login_redirect()
                except (UnicodeDecodeError, ValueError) as exc:
                    self._json({"error": str(exc)}, 400)
                return
            if path != "/api/save":
                self._json({"error": "Not found"}, 404)
                return
            if not self._authenticated() or not self._allowed_origin():
                self._json({"error": "Forbidden"}, 403)
                return
            if (
                self.headers.get("Content-Type", "").strip().lower()
                != "application/json"
            ):
                self._json({"error": "Content-Type must be application/json"}, 415)
                return
            try:
                raw_length = self.headers.get("Content-Length")
                length = int(raw_length) if raw_length is not None else 0
                if length <= 0 or length > MAX_PAYLOAD_BYTES:
                    raise ValueError("MCIF reliability-v2 payload size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != POST_KEYS:
                    raise ValueError("MCIF reliability-v2 POST payload keys differ")
                self._json(session.save(payload))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sheet", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--head-ledger", type=Path, required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument("--expected-role")
    parser.add_argument("--expected-stage")
    parser.add_argument("--identity-registry", type=Path, required=True)
    parser.add_argument("--expected-identity-registry-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--expected-run-contract-file-sha256", required=True)
    parser.add_argument("--access-token", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43874)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("MCIF reliability-v2 server must bind localhost")
    session = ReliabilityAnnotationSession(
        input_sheet=args.input_sheet,
        expected_input_sha256=args.expected_input_sha256,
        workspace_root=args.workspace_root,
        event_log=args.event_log,
        head_ledger=args.head_ledger,
        annotator_id=args.annotator_id,
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        hmac_key_path=args.hmac_key,
        expected_items=args.expected_items,
        expected_role=args.expected_role,
        expected_stage=args.expected_stage,
        identity_registry_path=args.identity_registry,
        expected_identity_registry_sha256=args.expected_identity_registry_sha256,
        run_contract_path=args.run_contract,
        expected_run_contract_file_sha256=args.expected_run_contract_file_sha256,
        access_token_path=args.access_token,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(
        f"MCIF reliability-v2 {session.role}/{session.stage}: "
        f"http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
