import copy
import hashlib
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

import pytest
from PIL import Image
from scripts.build_mcif_beyond_ocr_reliability_workspace import (
    TARGET_AUTHOR_SCHEMA,
    TARGET_VALIDATOR_STAGE1_SCHEMA,
    build_run_contract,
)
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256
from scripts.mcif_beyond_ocr_reliability import (
    IDENTITY_REGISTRY_SCHEMA,
    TARGET_ADJUDICATION_INPUT_SCHEMA,
    TARGET_VALIDATOR_STAGE2_SCHEMA,
    VISUAL_ADJUDICATION_INPUT_SCHEMA,
    VISUAL_FIELDS,
    VISUAL_INPUT_SCHEMAS,
    blank_annotation,
    create_access_token,
    create_hmac_key,
    expected_input_status,
    initialize_event_log,
    load_hmac_key,
    sign_release_row,
    write_jsonl_atomic,
)
from scripts.serve_mcif_beyond_ocr_reliability import (
    ReliabilityAnnotationSession,
    make_handler,
)

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "mcif_beyond_ocr_reliability_v2.json"
)


def _identity_registry():
    assignments = {
        "visual_a": "annotator-visual_a",
        "visual_b": "annotator-visual_b",
        "target_author": "annotator-target_author",
        "target_validator": "annotator-target_validator",
        "visual_adjudicator": "annotator-visual_adjudicator",
        "target_adjudicator": "annotator-target_adjudicator",
    }
    registry = {
        "schema_version": IDENTITY_REGISTRY_SCHEMA,
        "people": [
            {"person_id": person_id, "aliases": [f"{role}@example.test"]}
            for role, person_id in assignments.items()
        ],
        "role_assignments": assignments,
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    return registry


def _hashed(row):
    row = copy.deepcopy(row)
    row["row_sha256"] = canonical_sha256(row)
    return row


def _image(workspace: Path) -> dict:
    path = workspace / "media" / "slide.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), color=(22, 88, 130)).save(path)
    return {
        "path": "media/slide.png",
        "sha256": file_sha256(path),
        "width": 32,
        "height": 18,
    }


def _common(role: str, item_id: str, run_contract_sha256: str) -> dict:
    return {
        "status": "PENDING_TEST_ANNOTATION",
        "role": role,
        "item_id": item_id,
        "candidate_source_en": "the green system wins",
        "candidate_kind": "phrase",
        "candidate_token_count": 4,
        "annotation_status": "pending",
        "reason_codes": [],
        "annotation_note": "",
        "annotator_id": None,
        "locked_at_utc": None,
        "timing_exposed": False,
        "run_contract_sha256": run_contract_sha256,
    }


def _row(workspace: Path, role: str, stage: str, run_contract_sha256: str) -> dict:
    item_id = f"ITEM-{role}-{stage}"
    row = _common(role, item_id, run_contract_sha256)
    row["status"] = expected_input_status(role, stage)
    if role in {"visual_a", "visual_b"}:
        row.update(
            {
                "schema_version": VISUAL_INPUT_SCHEMAS[stage],
                "stage": stage,
                "r0_text": "green blue",
                VISUAL_FIELDS[stage]: None,
                "r1_exposed": stage in {"r1", "pixels", "descriptor"},
                "pixels_exposed": stage in {"pixels", "descriptor"},
                "descriptor_exposed": stage == "descriptor",
                "reference_exposed": False,
            }
        )
        if stage != "r0":
            row.update(
                {
                    "locked_judgments": {
                        VISUAL_FIELDS[prior]: "no"
                        for prior in ("r0", "r1", "pixels")
                        if ("r0", "r1", "pixels", "descriptor").index(prior)
                        < ("r0", "r1", "pixels", "descriptor").index(stage)
                    },
                    "prior_stage": "r0" if stage == "r1" else "r1",
                    "prior_input_row_sha256": "a" * 64,
                    "prior_freeze_hmac_sha256": "b" * 64,
                    "prior_cohort_lock_sha256": "c" * 64,
                }
            )
        if stage in {"r1", "pixels", "descriptor"}:
            row["r1_blocks"] = [
                {
                    "content_kind": "chart_markdown",
                    "label": "result",
                    "content": "green > blue",
                    "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                    "reading_order": 0,
                }
            ]
        if stage in {"pixels", "descriptor"}:
            row["current_slide"] = _image(workspace)
        if stage == "descriptor":
            row["proposed_evidence_origins"] = [
                {
                    "descriptor_field": "scene_summary",
                    "descriptor_index": 0,
                    "descriptor_text": "The green bar is taller.",
                    "descriptor_sha256": canonical_sha256("The green bar is taller."),
                }
            ]
        return _hashed(row)
    if role in {"target_author", "target_validator"}:
        row.update(
            {
                "source_reference_en": "The green system wins.",
                "target_reference_zh": "绿色系统获胜。",
                "slide_or_visual_exposed": False,
            }
        )
        if role == "target_author":
            row["schema_version"] = TARGET_AUTHOR_SCHEMA
            row.update(blank_annotation(role, stage))
        elif stage == "independent_alignment":
            row.update(
                {
                    "schema_version": TARGET_VALIDATOR_STAGE1_SCHEMA,
                    "stage": stage,
                    "candidate_eligibility": None,
                    "target_reference_alignment": None,
                    "author_identity_exposed": False,
                    "author_labels_exposed": False,
                    "author_scoring_text_exposed": False,
                }
            )
        else:
            row.update(
                {
                    "schema_version": TARGET_VALIDATOR_STAGE2_SCHEMA,
                    "stage": stage,
                    "locked_candidate_eligibility": "yes",
                    "locked_target_reference_alignment": "explicit",
                    "author_candidate_eligibility": "yes",
                    "author_canonical_source_event_en": "the green system wins",
                    "author_acceptable_target_realizations_zh": ["绿色系统获胜"],
                    "author_forbidden_target_realizations_zh": [],
                    "author_target_reference_alignment": "explicit",
                    "author_source_input_row_sha256": "a" * 64,
                    "author_freeze_hmac_sha256": "b" * 64,
                    "validator_stage1_input_row_sha256": "c" * 64,
                    "validator_stage1_freeze_hmac_sha256": "d" * 64,
                    "review_decision": None,
                    "edited_canonical_source_event_en": "",
                    "edited_acceptable_target_realizations_zh": [],
                    "edited_forbidden_target_realizations_zh": [],
                    "author_identity_exposed": False,
                }
            )
        return _hashed(row)
    if role == "visual_adjudicator":
        row.update(
            {
                "schema_version": VISUAL_ADJUDICATION_INPUT_SCHEMA,
                "stage": stage,
                "primitive_field": "pixel_support",
                "released_evidence": {
                    "r0_text": "green blue",
                    "r1_blocks": [
                        {
                            "content_kind": "chart_markdown",
                            "label": "result",
                            "content": "green > blue",
                            "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                            "reading_order": 0,
                        }
                    ],
                    "current_slide": _image(workspace),
                },
                "visual_a_raw": {
                    "judgment": "yes",
                    "reason_codes": [],
                    "note": "visible",
                },
                "visual_b_raw": {
                    "judgment": "no",
                    "reason_codes": ["ambiguous_visual"],
                    "note": "unclear",
                },
                "pre_adjudication_row_sha256": "e" * 64,
                "adjudicated_judgment": None,
                "reference_exposed": False,
            }
        )
        return _hashed(row)
    row.update(
        {
            "schema_version": TARGET_ADJUDICATION_INPUT_SCHEMA,
            "stage": stage,
            "released_source": {
                "source_reference_en": "The green system wins.",
                "target_reference_zh": "绿色系统获胜。",
            },
            "target_raw": {
                "candidate_eligibility": {"author": "yes", "validator": "no"},
                "target_reference_alignment": {
                    "author": "explicit",
                    "validator": "unsupported",
                },
                "stage2_review_decision": "reject",
                "author_reason_codes": [],
                "validator_stage1_reason_codes": ["target_reference_mismatch"],
                "validator_stage2_reason_codes": ["canonical_event_incorrect"],
            },
            "author_scoring_text": {
                "canonical_source_event_en": "the green system wins",
                "acceptable_target_realizations_zh": ["绿色系统获胜"],
                "forbidden_target_realizations_zh": [],
            },
            "validator_edits": {
                "canonical_source_event_en": "",
                "acceptable_target_realizations_zh": [],
                "forbidden_target_realizations_zh": [],
            },
            "pre_adjudication_row_sha256": "f" * 64,
            "adjudication_decision": None,
            "final_canonical_source_event_en": "",
            "final_acceptable_target_realizations_zh": [],
            "final_forbidden_target_realizations_zh": [],
            "slide_or_visual_exposed": False,
        }
    )
    return _hashed(row)


def _session(tmp_path: Path, role: str = "visual_a", stage: str = "r0"):
    workspace = tmp_path / f"workspace-{role}-{stage}"
    workspace.mkdir(parents=True)
    key_path = tmp_path / f"{role}-{stage}.key"
    create_hmac_key(key_path)
    key = load_hmac_key(key_path)
    access_token_path = tmp_path / f"{role}-{stage}.access-token"
    create_access_token(access_token_path)
    registry_payload = _identity_registry()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    role_access_token_sha256 = {
        required_role: hashlib.sha256(
            f"unused-access-token-{required_role}".encode()
        ).hexdigest()
        for required_role in config["identity"]["required_disjoint_roles"]
    }
    role_access_token_sha256[role] = file_sha256(access_token_path)
    run_contract = build_run_contract(
        config=config,
        config_file_sha256=file_sha256(CONFIG_PATH),
        identity_registry=registry_payload,
        release_key=key,
        expected_items=1,
        expected_visual_sha256="1" * 64,
        expected_target_sha256="2" * 64,
        expected_mapping_sha256="3" * 64,
        private_visual_rows_sha256="6" * 64,
        private_target_rows_sha256="7" * 64,
        private_mapping_rows_sha256="8" * 64,
        role_access_token_sha256=role_access_token_sha256,
        source_hf_revision="4" * 40,
        builder_git_commit="5" * 40,
    )
    run_contract_sha256 = canonical_sha256(run_contract)
    row = sign_release_row(_row(workspace, role, stage, run_contract_sha256), key)
    input_sheet = workspace / "items.jsonl"
    write_jsonl_atomic(input_sheet, [row])
    event_log = tmp_path / f"{role}-{stage}.events.jsonl"
    head_ledger = tmp_path / f"{role}-{stage}.event-heads.jsonl"
    initialize_event_log(
        event_log,
        head_ledger,
        [row],
        annotator_id=f"annotator-{role}",
        expected_items=1,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    identity_registry = tmp_path / f"{role}-{stage}.identity-registry.json"
    identity_registry.write_text(
        json.dumps(registry_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_contract_path = tmp_path / f"{role}-{stage}.run-contract.json"
    run_contract_path.write_text(
        json.dumps(run_contract, sort_keys=True) + "\n", encoding="utf-8"
    )
    kwargs = {
        "input_sheet": input_sheet,
        "expected_input_sha256": file_sha256(input_sheet),
        "workspace_root": workspace,
        "event_log": event_log,
        "head_ledger": head_ledger,
        "annotator_id": f"annotator-{role}",
        "config_path": CONFIG_PATH,
        "expected_config_sha256": file_sha256(CONFIG_PATH),
        "hmac_key_path": key_path,
        "expected_items": 1,
        "expected_role": role,
        "expected_stage": stage,
        "identity_registry_path": identity_registry,
        "expected_identity_registry_sha256": file_sha256(identity_registry),
        "run_contract_path": run_contract_path,
        "expected_run_contract_file_sha256": file_sha256(run_contract_path),
        "access_token_path": access_token_path,
    }
    return ReliabilityAnnotationSession(**kwargs), kwargs


def _visual_annotation(value="yes"):
    return {
        "r0_support": value,
        "reason_codes": [] if value == "yes" else ["ambiguous_visual"],
        "annotation_note": "",
    }


def _completed_annotation(role: str, stage: str):
    annotation = blank_annotation(role, stage)
    if role in {"visual_a", "visual_b"}:
        annotation[VISUAL_FIELDS[stage]] = "yes"
    elif role == "target_author":
        annotation.update(
            {
                "candidate_eligibility": "yes",
                "canonical_source_event_en": "the green system wins",
                "acceptable_target_realizations_zh": ["绿色系统获胜"],
                "target_reference_alignment": "explicit",
            }
        )
    elif role == "target_validator" and stage == "independent_alignment":
        annotation.update(
            {
                "candidate_eligibility": "yes",
                "target_reference_alignment": "explicit",
            }
        )
    elif role == "target_validator":
        annotation["review_decision"] = "accept"
    elif role == "visual_adjudicator":
        annotation.update(
            {
                "adjudicated_judgment": "yes",
                "reason_codes": ["ambiguous_visual"],
            }
        )
    else:
        annotation.update(
            {
                "adjudication_decision": "accept",
                "final_canonical_source_event_en": "the green system wins",
                "final_acceptable_target_realizations_zh": ["绿色系统获胜"],
                "reason_codes": ["ambiguous_alignment"],
            }
        )
    return annotation


def _request(server, method, path, *, body=None, headers=None):
    connection = HTTPConnection(*server.server_address, timeout=2)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def _login(server, session, access_token_path: Path):
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    body = urlencode(
        {"access_token": access_token_path.read_text(encoding="ascii")}
    ).encode()
    status, headers, _ = _request(
        server,
        "POST",
        "/login",
        body=body,
        headers={
            "Origin": origin,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert status == 303
    assert headers["Location"] == "/"
    assert "HttpOnly" in headers["Set-Cookie"]
    assert "SameSite=Strict" in headers["Set-Cookie"]
    assert session.cookie_matches(headers["Set-Cookie"])
    return headers["Set-Cookie"].split(";", 1)[0]


@pytest.fixture
def live_server(tmp_path):
    session, _ = _session(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, session
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_response_projection_removes_hashes_locks_and_bbox_fields(tmp_path):
    session, _ = _session(tmp_path, "visual_a", "r1")
    state = session.state(0)
    encoded = json.dumps(state, ensure_ascii=False)
    assert "bbox_norm" not in encoded
    assert "row_sha256" not in encoded
    assert "prior_freeze_hmac_sha256" not in encoded
    assert "prior_cohort_lock_sha256" not in encoded
    assert "r1_blocks" in state["item"]
    assert "image_available" not in state["item"]
    assert "proposed_evidence_origins" not in state["item"]


def test_http_cookie_origin_content_type_and_exact_payload(live_server):
    server, session = live_server
    status, headers, _ = _request(server, "GET", "/")
    assert status == 200
    assert "Set-Cookie" not in headers
    assert _request(server, "GET", "/api/meta")[0] == 403
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    assert (
        _request(
            server,
            "POST",
            "/login",
            body=b"access_token=wrong",
            headers={
                "Origin": origin,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )[0]
        == 403
    )
    cookie = _login(server, session, session.access_token_path)
    body = json.dumps(
        {
            "item_id": session.state(0)["item"]["item_id"],
            "expected_event_index": 0,
            "annotation_status": "completed",
            "annotation": _visual_annotation(),
        }
    ).encode()
    valid_headers = {
        "Cookie": cookie,
        "Origin": origin,
        "Content-Type": "application/json",
    }
    assert _request(server, "POST", "/api/save", body=body)[0] == 403
    assert (
        _request(
            server,
            "POST",
            "/api/save",
            body=body,
            headers={**valid_headers, "Origin": "http://evil.invalid"},
        )[0]
        == 403
    )
    assert (
        _request(
            server,
            "POST",
            "/api/save",
            body=body,
            headers={**valid_headers, "Content-Type": "text/plain"},
        )[0]
        == 415
    )
    payload_with_extra = json.loads(body)
    payload_with_extra["submitted_at_utc"] = "attacker-controlled"
    assert (
        _request(
            server,
            "POST",
            "/api/save",
            body=json.dumps(payload_with_extra).encode(),
            headers=valid_headers,
        )[0]
        == 400
    )
    status, _, response = _request(
        server, "POST", "/api/save", body=body, headers=valid_headers
    )
    assert status == 200
    saved = json.loads(response)
    assert saved["state"]["item"]["annotation_status"] == "completed"


def test_stale_and_completed_overwrite_are_rejected_append_only(tmp_path):
    session, kwargs = _session(tmp_path)
    second_process_view = ReliabilityAnnotationSession(**kwargs)
    payload = {
        "item_id": session.state(0)["item"]["item_id"],
        "expected_event_index": 0,
        "annotation_status": "draft",
        "annotation": _visual_annotation(),
    }
    assert session.save(payload)["state"]["item"]["event_index"] == 1
    with pytest.raises(ValueError, match="stale event version"):
        second_process_view.save(payload)
    payload["expected_event_index"] = 1
    payload["annotation_status"] = "completed"
    assert session.save(payload)["state"]["item"]["event_index"] == 2
    payload["expected_event_index"] = 2
    with pytest.raises(ValueError, match="completed annotation is immutable"):
        session.save(payload)
    assert len(kwargs["event_log"].read_text(encoding="utf-8").splitlines()) == 3


def test_startup_validates_full_event_log_signature(tmp_path):
    _, kwargs = _session(tmp_path)
    events = [json.loads(line) for line in kwargs["event_log"].read_text().splitlines()]
    events[0]["annotation_note"] = "tampered"
    write_jsonl_atomic(kwargs["event_log"], events)
    with pytest.raises(ValueError, match="event signature differs"):
        ReliabilityAnnotationSession(**kwargs)


def test_startup_requires_the_registered_canonical_role_identity(tmp_path):
    _, kwargs = _session(tmp_path)
    with pytest.raises(ValueError, match="annotator differs from identity registry"):
        ReliabilityAnnotationSession(
            **{**kwargs, "annotator_id": "visual_a@example.test"}
        )


def test_startup_rejects_access_token_swap_and_permission_drift(tmp_path):
    _, kwargs = _session(tmp_path)
    swapped_token = tmp_path / "swapped.access-token"
    create_access_token(swapped_token)
    with pytest.raises(ValueError, match="access token differs from run contract"):
        ReliabilityAnnotationSession(**{**kwargs, "access_token_path": swapped_token})

    kwargs["access_token_path"].chmod(0o640)
    with pytest.raises(ValueError, match="access token permissions must be 0600"):
        ReliabilityAnnotationSession(**kwargs)


def test_media_is_item_bound_rechecked_and_rejects_traversal_or_bad_hash(tmp_path):
    session, kwargs = _session(tmp_path, "visual_a", "pixels")
    item_id = session.state(0)["item"]["item_id"]
    payload, content_type = session.media_bytes(item_id)
    assert payload
    assert content_type == "image/png"
    media_path = kwargs["workspace_root"] / "media" / "slide.png"
    Image.new("RGB", (32, 18), color=(240, 20, 20)).save(media_path)
    with pytest.raises(ValueError, match="image hash differs"):
        session.media_bytes(item_id)

    traversal_root = tmp_path / "traversal"
    traversal_root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (32, 18), color=(1, 2, 3)).save(outside)
    row = _row(traversal_root, "visual_a", "pixels", session.run_contract_sha256)
    row["current_slide"] = {
        "path": "../outside.png",
        "sha256": file_sha256(outside),
        "width": 32,
        "height": 18,
    }
    key = session.key
    row = sign_release_row(
        _hashed(
            {
                key_name: value
                for key_name, value in row.items()
                if key_name not in {"row_sha256", "release_hmac_sha256"}
            }
        ),
        key,
    )
    traversal_input = traversal_root / "items.jsonl"
    write_jsonl_atomic(traversal_input, [row])
    event_log = tmp_path / "traversal.events.jsonl"
    head_ledger = tmp_path / "traversal.event-heads.jsonl"
    initialize_event_log(
        event_log,
        head_ledger,
        [row],
        annotator_id="annotator-visual_a",
        expected_items=1,
        key=key,
        config=session.config,
        run_contract_sha256=session.run_contract_sha256,
    )
    traversal_kwargs = {
        **kwargs,
        "input_sheet": traversal_input,
        "expected_input_sha256": file_sha256(traversal_input),
        "workspace_root": traversal_root,
        "event_log": event_log,
        "head_ledger": head_ledger,
    }
    with pytest.raises(ValueError, match="canonical and relative"):
        ReliabilityAnnotationSession(**traversal_kwargs)

    bad_hash_row = copy.deepcopy(row)
    bad_hash_row["current_slide"]["path"] = "media/slide.png"
    (traversal_root / "media").mkdir(exist_ok=True)
    Image.new("RGB", (32, 18), color=(1, 2, 3)).save(
        traversal_root / "media" / "slide.png"
    )
    bad_hash_row["current_slide"]["sha256"] = "0" * 64
    bad_hash_row = sign_release_row(
        _hashed(
            {
                key_name: value
                for key_name, value in bad_hash_row.items()
                if key_name not in {"row_sha256", "release_hmac_sha256"}
            }
        ),
        key,
    )
    bad_input = traversal_root / "bad-hash.jsonl"
    write_jsonl_atomic(bad_input, [bad_hash_row])
    bad_events = tmp_path / "bad-hash.events.jsonl"
    bad_heads = tmp_path / "bad-hash.event-heads.jsonl"
    initialize_event_log(
        bad_events,
        bad_heads,
        [bad_hash_row],
        annotator_id="annotator-visual_a",
        expected_items=1,
        key=key,
        config=session.config,
        run_contract_sha256=session.run_contract_sha256,
    )
    with pytest.raises(ValueError, match="image hash differs"):
        ReliabilityAnnotationSession(
            **{
                **traversal_kwargs,
                "input_sheet": bad_input,
                "expected_input_sha256": file_sha256(bad_input),
                "event_log": bad_events,
                "head_ledger": bad_heads,
            }
        )


@pytest.mark.parametrize(
    ("role", "stage"),
    [
        ("visual_a", "r0"),
        ("visual_b", "r1"),
        ("visual_a", "pixels"),
        ("visual_b", "descriptor"),
        ("target_author", "author"),
        ("target_validator", "independent_alignment"),
        ("target_validator", "author_text_review"),
        ("visual_adjudicator", "visual_resolution"),
        ("target_adjudicator", "target_resolution"),
    ],
)
def test_all_role_stage_schemas_have_explicit_projection_and_blank_annotation(
    tmp_path, role, stage
):
    session, _ = _session(tmp_path, role, stage)
    state = session.state(0)
    assert state["role"] == role
    assert state["stage"] == stage
    assert state["item"]["annotation"] == blank_annotation(role, stage)
    encoded = json.dumps(state, ensure_ascii=False)
    assert "row_sha256" not in encoded
    assert "event_hmac_sha256" not in encoded
    assert "bbox_norm" not in encoded
    saved = session.save(
        {
            "item_id": state["item"]["item_id"],
            "expected_event_index": 0,
            "annotation_status": "completed",
            "annotation": _completed_annotation(role, stage),
        }
    )
    assert saved["state"]["item"]["annotation_status"] == "completed"
