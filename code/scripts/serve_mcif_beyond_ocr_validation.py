#!/usr/bin/env python3
"""Serve localhost-only MCIF beyond-OCR role-specific annotation UI."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    load_jsonl,
    resolve_regular_file,
)
from scripts.mcif_beyond_ocr_validation import (
    initialize_working_rows,
    load_frozen_config,
    validate_input_rows,
    validate_working_row,
    validate_working_rows,
    write_jsonl_atomic,
)


VISUAL_SAVE_FIELDS = {
    "annotation_status",
    "visual_evidence_correct",
    "candidate_supported_by_visual_evidence",
    "r0_insufficient",
    "r1_insufficient",
    "reason_codes",
    "annotation_note",
}
TARGET_SAVE_FIELDS = {
    "annotation_status",
    "candidate_eligibility",
    "canonical_source_event_en",
    "acceptable_target_realizations_zh",
    "forbidden_target_realizations_zh",
    "target_reference_alignment",
    "reason_codes",
    "annotation_note",
}


class BeyondOcrValidationSession:
    def __init__(
        self,
        *,
        role: str,
        input_sheet: Path,
        expected_input_sha256: str,
        workspace_root: Path,
        working_sheet: Path,
        annotator_id: str,
        config_path: Path,
        expected_config_sha256: str,
        expected_items: int,
    ) -> None:
        load_frozen_config(config_path, expected_config_sha256)
        if file_sha256(input_sheet) != expected_input_sha256:
            raise ValueError(f"MCIF beyond-OCR {role} UI input hash differs")
        if workspace_root.is_symlink():
            raise ValueError("MCIF beyond-OCR UI workspace root cannot be a symlink")
        self.workspace_root = workspace_root.resolve(strict=True)
        resolved_input = input_sheet.resolve(strict=True)
        if not resolved_input.is_file() or not resolved_input.is_relative_to(
            self.workspace_root
        ):
            raise ValueError("MCIF beyond-OCR UI input is outside its role workspace")
        self.role = role
        self.input_rows = load_jsonl(input_sheet)
        self.input_by_id = validate_input_rows(
            self.input_rows, role=role, expected_items=expected_items
        )
        self.working_sheet = working_sheet
        self.annotator_id = annotator_id
        self.expected_items = expected_items
        if working_sheet.exists() or working_sheet.is_symlink():
            if working_sheet.is_symlink() or not working_sheet.is_file():
                raise ValueError("MCIF beyond-OCR UI working sheet is not a regular file")
            self.working_rows = load_jsonl(working_sheet)
            validate_working_rows(
                self.working_rows,
                self.input_rows,
                role=role,
                annotator_id=annotator_id,
                expected_items=expected_items,
                allow_pending=True,
            )
        else:
            self.working_rows = initialize_working_rows(
                self.input_rows,
                role=role,
                annotator_id=annotator_id,
                expected_items=expected_items,
            )
            write_jsonl_atomic(working_sheet, self.working_rows)
        self.media_by_path = {}
        if role == "visual":
            for row in self.input_rows:
                relative = row["current_slide"]["path"]
                media = resolve_regular_file(self.workspace_root, relative)
                if file_sha256(media) != row["current_slide"]["sha256"]:
                    raise ValueError("MCIF beyond-OCR UI media hash differs")
                self.media_by_path[relative] = media

    def progress(self) -> dict[str, int]:
        completed = sum(row["annotation_status"] == "completed" for row in self.working_rows)
        return {"completed": completed, "total": len(self.working_rows)}

    def state(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.working_rows):
            raise ValueError("MCIF beyond-OCR UI index is outside the sheet")
        return {
            "role": self.role,
            "index": index,
            "item": self.working_rows[index],
            "progress": self.progress(),
        }

    def next_pending(self, current: int) -> int:
        for offset in range(1, len(self.working_rows) + 1):
            index = (current + offset) % len(self.working_rows)
            if self.working_rows[index]["annotation_status"] == "pending":
                return index
        return min(current + 1, len(self.working_rows) - 1)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) - ({"index"} | (VISUAL_SAVE_FIELDS if self.role == "visual" else TARGET_SAVE_FIELDS)):
            raise ValueError("MCIF beyond-OCR UI payload contains forbidden fields")
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("MCIF beyond-OCR UI payload has an invalid index")
        source = self.input_rows[index] if 0 <= index < len(self.input_rows) else None
        if source is None:
            raise ValueError("MCIF beyond-OCR UI index is outside the sheet")
        current = self.working_rows[index]
        updated = dict(current)
        allowed = VISUAL_SAVE_FIELDS if self.role == "visual" else TARGET_SAVE_FIELDS
        for key in allowed:
            if key in payload:
                updated[key] = payload[key]
        updated["row_sha256"] = canonical_sha256(
            {key: value for key, value in updated.items() if key != "row_sha256"}
        )
        validate_working_row(
            updated,
            source,
            role=self.role,
            annotator_id=self.annotator_id,
            allow_pending=True,
        )
        replacement = list(self.working_rows)
        replacement[index] = updated
        validate_working_rows(
            replacement,
            self.input_rows,
            role=self.role,
            annotator_id=self.annotator_id,
            expected_items=self.expected_items,
            allow_pending=True,
        )
        write_jsonl_atomic(self.working_sheet, replacement)
        self.working_rows = replacement
        return updated

    def media(self, relative: str) -> Path:
        if self.role != "visual" or relative not in self.media_by_path:
            raise FileNotFoundError("MCIF beyond-OCR media is not in the visual view")
        return self.media_by_path[relative]


def page_html(role: str) -> str:
    visual = role == "visual"
    role_title = "Visual evidence validation" if visual else "Target event authoring"
    role_markup = (
        """
        <main class="visual-layout">
          <section class="slide-band"><img id="slide" alt="Current slide"></section>
          <section class="evidence-band">
            <div class="candidate"><span>Candidate</span><strong id="candidate"></strong><small id="channel"></small></div>
            <div class="evidence-grid">
              <article><h2>Proposed evidence</h2><div id="origins"></div></article>
              <article><h2>R0 flat OCR</h2><pre id="r0"></pre></article>
              <article><h2>R1 structured blocks</h2><div id="r1"></div></article>
            </div>
          </section>
          <section class="form-band" id="visual-form">
            <div class="judgments" id="visual-judgments"></div>
            <div class="reasons" id="visual-reasons"></div>
            <label>Note<textarea id="note" rows="3"></textarea></label>
          </section>
        </main>
        """
        if visual
        else """
        <main class="target-layout">
          <section class="candidate"><span>Candidate</span><strong id="candidate"></strong></section>
          <section class="reference-grid">
            <article><h2>English source</h2><p id="source"></p></article>
            <article><h2>Chinese reference</h2><p id="target"></p></article>
          </section>
          <section class="form-band" id="target-form">
            <div class="judgments" id="target-judgments"></div>
            <div class="text-grid">
              <label>Canonical English event<input id="canonical" type="text"></label>
              <label>Target alignment<select id="alignment"><option value="">Select</option><option>explicit</option><option>paraphrased</option><option>omitted</option><option>unsupported</option><option>uncertain</option></select></label>
              <label>Acceptable Chinese realizations<textarea id="acceptable" rows="3"></textarea></label>
              <label>Forbidden Chinese realizations<textarea id="forbidden" rows="3"></textarea></label>
            </div>
            <div class="reasons" id="target-reasons"></div>
            <label>Note<textarea id="note" rows="3"></textarea></label>
          </section>
        </main>
        """
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCIF {role_title}</title>
<style>
:root{{--bg:#f7f8fa;--ink:#18202a;--muted:#5e6875;--line:#d7dce2;--paper:#fff;--accent:#0b6b58;--warn:#9b4c12}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;display:grid;grid-template-columns:1fr auto auto;gap:16px;align-items:center;padding:12px 22px;background:var(--paper);border-bottom:1px solid var(--line)}}
h1{{font-size:17px;margin:0}} .progress{{font-variant-numeric:tabular-nums;color:var(--muted)}} .nav{{display:flex;gap:6px}}
button{{height:36px;min-width:36px;border:1px solid var(--line);background:#fff;color:var(--ink);font:inherit;cursor:pointer}} button:hover{{border-color:#82909f}} button.primary{{background:var(--accent);border-color:var(--accent);color:white;padding:0 16px}}
.visual-layout,.target-layout{{max-width:1440px;margin:0 auto}} .slide-band{{background:#20252b;padding:18px;display:flex;justify-content:center}}
.slide-band img{{display:block;max-width:100%;width:auto;height:auto;max-height:62vh;object-fit:contain}}
.candidate{{display:flex;align-items:baseline;gap:12px;padding:16px 22px;background:#fff;border-bottom:1px solid var(--line)}}
.candidate span,.candidate small{{color:var(--muted)}} .candidate strong{{font-size:20px}} .candidate small{{margin-left:auto}}
.evidence-grid,.reference-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-bottom:1px solid var(--line)}}
.reference-grid{{grid-template-columns:1fr 1fr}} article{{padding:18px 22px;min-width:0;background:#fff;border-right:1px solid var(--line)}} article:last-child{{border-right:0}}
h2{{font-size:13px;text-transform:uppercase;color:var(--muted);margin:0 0 10px}} pre{{white-space:pre-wrap;word-break:break-word;margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}
.origin,.block{{padding:8px 0;border-bottom:1px solid #edf0f2}} .origin:last-child,.block:last-child{{border-bottom:0}} .meta{{font-size:12px;color:var(--muted);margin-bottom:3px}}
.form-band{{padding:20px 22px 100px}} .judgments{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:18px}}
.judgment{{background:#fff;border:1px solid var(--line);padding:14px}} .judgment>span{{display:block;font-weight:600;margin-bottom:10px}}
.segments{{display:grid;grid-template-columns:repeat(3,1fr)}} .segments label{{position:relative}} .segments input{{position:absolute;opacity:0}}
.segments b{{display:block;text-align:center;padding:8px;border:1px solid var(--line);border-right:0;font-weight:500;cursor:pointer}} .segments label:last-child b{{border-right:1px solid var(--line)}}
.segments input:checked+b{{background:#dceee9;border-color:var(--accent);color:#074b3e}}
.reasons{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}} .reasons label{{background:#fff;border:1px solid var(--line);padding:7px 10px;cursor:pointer}}
.reasons input{{margin:0 6px 0 0}} .text-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px}}
label{{display:block;font-weight:600}} input[type=text],select,textarea{{display:block;width:100%;margin-top:6px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:9px;font:inherit;border-radius:0}} textarea{{resize:vertical}}
.footer{{position:fixed;left:0;right:0;bottom:0;display:flex;justify-content:space-between;align-items:center;padding:12px 22px;background:#fff;border-top:1px solid var(--line)}}
.status{{color:var(--muted)}} .error{{color:#a43124}}
@media(max-width:800px){{header{{grid-template-columns:1fr auto;padding:10px 12px}}h1{{font-size:15px}}.progress{{grid-row:2}}.slide-band{{padding:8px}}.slide-band img{{max-height:45vh}}.evidence-grid,.reference-grid,.judgments,.text-grid{{grid-template-columns:1fr}}article{{border-right:0;border-bottom:1px solid var(--line);padding:14px}}.form-band,.candidate{{padding-left:14px;padding-right:14px}}.candidate{{align-items:flex-start;flex-wrap:wrap}}.candidate small{{margin-left:0;width:100%}}}}
</style></head><body>
<header><h1>{role_title}</h1><div class="progress" id="progress">0 / 0</div><div class="nav"><button id="prev" title="Previous">←</button><button id="next" title="Next">→</button></div></header>
{role_markup}
<div class="footer"><span class="status" id="status"></span><button class="primary" id="save" title="Save and advance">Save</button></div>
<script>
const ROLE={json.dumps(role)}; let state=null;
const judgmentLabels={{visual_evidence_correct:'Evidence description is correct',candidate_supported_by_visual_evidence:'Candidate is visually supported',r0_insufficient:'R0 flat OCR is insufficient',r1_insufficient:'R1 structured text is insufficient',candidate_eligibility:'Candidate is a scoreable target event'}};
const reasons=ROLE==='visual'?['incorrect_description','candidate_not_visible','r0_already_sufficient','r1_already_sufficient','generic_or_ambiguous','other']:['generic_or_unscorable','no_target_alignment','wrong_candidate_scope','reference_quality','other'];
function esc(x){{return String(x??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]))}}
function judgment(key,value){{return `<div class="judgment"><span>${{judgmentLabels[key]}}</span><div class="segments">${{['yes','no','uncertain'].map(v=>`<label><input type="radio" name="${{key}}" value="${{v}}" ${{value===v?'checked':''}}><b>${{v}}</b></label>`).join('')}}</div></div>`}}
function reasonMarkup(selected){{return reasons.map(v=>`<label><input type="checkbox" name="reason" value="${{v}}" ${{selected.includes(v)?'checked':''}}>${{v.replaceAll('_',' ')}}</label>`).join('')}}
function lines(x){{return String(x||'').split(/\\n/).map(v=>v.trim()).filter(Boolean)}}
async function load(index){{const r=await fetch(`/api/state?index=${{index}}`); const data=await r.json(); if(!r.ok)throw Error(data.error); state=data; render()}}
function render(){{const x=state.item;document.getElementById('progress').textContent=`${{state.progress.completed}} / ${{state.progress.total}}`;document.getElementById('candidate').textContent=x.candidate_source_en;document.getElementById('note').value=x.annotation_note||'';document.getElementById('status').textContent=`${{x.item_id}} · ${{x.annotation_status}}`;
if(ROLE==='visual'){{document.getElementById('slide').src='/' + x.current_slide.path;document.getElementById('channel').textContent=x.evidence_channel.replaceAll('_',' ');document.getElementById('r0').textContent=x.current_slide_r0_text;document.getElementById('origins').innerHTML=x.proposed_evidence_origins.map(o=>`<div class="origin"><div class="meta">${{esc(o.descriptor_field||o.content_kind||'evidence')}}</div>${{esc(o.descriptor_text||o.content||'')}}</div>`).join('');document.getElementById('r1').innerHTML=x.current_slide_r1_blocks.map(b=>`<div class="block"><div class="meta">${{esc(b.content_kind)}} · ${{esc(b.label)}}</div>${{esc(b.content)}}</div>`).join('');let keys=['visual_evidence_correct','candidate_supported_by_visual_evidence','r0_insufficient'];if(x.requires_r1_insufficiency_judgment)keys.push('r1_insufficient');document.getElementById('visual-judgments').innerHTML=keys.map(k=>judgment(k,x[k])).join('');document.getElementById('visual-reasons').innerHTML=reasonMarkup(x.reason_codes||[])}}
else{{document.getElementById('source').textContent=x.source_reference_en;document.getElementById('target').textContent=x.target_reference_zh;document.getElementById('target-judgments').innerHTML=judgment('candidate_eligibility',x.candidate_eligibility);document.getElementById('canonical').value=x.canonical_source_event_en||'';document.getElementById('alignment').value=x.target_reference_alignment||'';document.getElementById('acceptable').value=(x.acceptable_target_realizations_zh||[]).join('\\n');document.getElementById('forbidden').value=(x.forbidden_target_realizations_zh||[]).join('\\n');document.getElementById('target-reasons').innerHTML=reasonMarkup(x.reason_codes||[])}}
}}
function value(name){{return document.querySelector(`input[name="${{name}}"]:checked`)?.value||null}} function selectedReasons(){{return [...document.querySelectorAll('input[name="reason"]:checked')].map(x=>x.value)}}
async function save(){{let p={{index:state.index,annotation_status:'completed',annotation_note:document.getElementById('note').value,reason_codes:selectedReasons()}};if(ROLE==='visual'){{p.visual_evidence_correct=value('visual_evidence_correct');p.candidate_supported_by_visual_evidence=value('candidate_supported_by_visual_evidence');p.r0_insufficient=value('r0_insufficient');p.r1_insufficient=state.item.requires_r1_insufficiency_judgment?value('r1_insufficient'):null}}else{{p.candidate_eligibility=value('candidate_eligibility');p.canonical_source_event_en=document.getElementById('canonical').value;p.acceptable_target_realizations_zh=lines(document.getElementById('acceptable').value);p.forbidden_target_realizations_zh=lines(document.getElementById('forbidden').value);p.target_reference_alignment=document.getElementById('alignment').value||null}}const r=await fetch('/api/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(p)}});const data=await r.json();if(!r.ok)throw Error(data.error);await load(data.next_index)}}
document.getElementById('prev').onclick=()=>load(Math.max(0,state.index-1)).catch(show);document.getElementById('next').onclick=()=>load(Math.min(state.progress.total-1,state.index+1)).catch(show);document.getElementById('save').onclick=()=>save().catch(show);function show(e){{const s=document.getElementById('status');s.textContent=e.message;s.className='status error'}}load(0).catch(show);
</script></body></html>"""


def make_handler(session: BeyondOcrValidationSession):
    page = page_html(session.role).encode()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if parsed.path == "/api/state":
                    values = parse_qs(parsed.query).get("index", ["0"])
                    self.send_json(session.state(int(values[0])))
                    return
                relative = unquote(parsed.path.lstrip("/"))
                media = session.media(relative)
                body = media.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(media.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

        def do_POST(self) -> None:
            try:
                if urlparse(self.path).path != "/api/save":
                    self.send_json({"error": "Not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("MCIF beyond-OCR UI payload size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("MCIF beyond-OCR UI payload must be an object")
                item = session.save(payload)
                self.send_json(
                    {"item": item, "next_index": session.next_pending(payload["index"])}
                )
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("visual", "target"), required=True)
    parser.add_argument("--input-sheet", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--working-sheet", type=Path, required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-items", type=int, default=152)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("MCIF beyond-OCR validation server must bind localhost")
    session = BeyondOcrValidationSession(
        role=args.role,
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
    print(f"MCIF beyond-OCR {args.role}: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
