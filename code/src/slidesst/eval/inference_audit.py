from __future__ import annotations

import hashlib
import json
import posixpath
import shlex
import socket
import subprocess
from datetime import datetime, timezone
from typing import Callable, Sequence

from slidesst.eval.event_timing import (
    InferenceEnvironmentAudit,
    canonical_absolute_posix_path,
    command_contains_exact_marker,
    worker_process_identity_tree_sha256,
)


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def capture_inference_environment(
    *,
    run_id: str,
    container_name: str,
    worker_command_match: str,
    inference_repo_path: str,
    forbidden_container_artifact_roots: Sequence[str],
    forbidden_host_mount_source_roots: Sequence[str],
    capture_phase: str,
    run_command: RunCommand = subprocess.run,
    capture_host: str | None = None,
    captured_at_utc: str | None = None,
) -> InferenceEnvironmentAudit:
    if capture_phase not in {"workers_start", "workers_end"}:
        raise ValueError("capture phase must be workers_start or workers_end")
    if not worker_command_match.strip():
        raise ValueError("worker command match cannot be empty")
    try:
        marker_tokens = shlex.split(worker_command_match)
    except ValueError as exc:
        raise ValueError("worker command match is not valid shell-token syntax") from exc
    if marker_tokens.count(run_id) != 1:
        raise ValueError("worker command match must contain the exact run id token once")
    inspect = run_command(
        ["docker", "inspect", container_name],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_rows = json.loads(inspect.stdout)
    if len(inspect_rows) != 1:
        raise ValueError("docker inspect must return exactly one container")
    container = inspect_rows[0]
    if not container.get("State", {}).get("Running", False):
        raise ValueError("inference container is not running")

    mounts = sorted(
        [
            {
                "source": mount["Source"],
                "destination": mount["Destination"],
                "read_only": not bool(mount["RW"]),
            }
            for mount in container.get("Mounts", [])
        ],
        key=lambda mount: (mount["destination"], mount["source"]),
    )
    inference_repo_path = run_command(
        ["docker", "exec", container_name, "readlink", "-f", inference_repo_path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    normalized_container_roots = [
        run_command(
            ["docker", "exec", container_name, "realpath", "-m", root],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for root in forbidden_container_artifact_roots
    ]
    normalized_host_roots = [posixpath.normpath(root) for root in forbidden_host_mount_source_roots]
    if any(not canonical_absolute_posix_path(root) for root in normalized_host_roots):
        raise ValueError("forbidden host mount source roots must be canonical absolute paths")
    process_listing = run_command(
        ["docker", "exec", container_name, "ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    process_rows: dict[int, tuple[int, str]] = {}
    for line in process_listing.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        process_rows[int(fields[0])] = (int(fields[1]), fields[2])
    marker_pids = {
        pid
        for pid, (_, command) in process_rows.items()
        if command_contains_exact_marker(command, worker_command_match)
    }
    if not marker_pids:
        raise ValueError("no running inference workers matched the frozen command marker")
    related_pids = set(marker_pids)
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in process_rows.items():
            if pid not in related_pids and parent_pid in related_pids:
                related_pids.add(pid)
                changed = True

    workers: list[dict] = []
    for pid in sorted(related_pids):
        parent_pid, command = process_rows[pid]
        process_stat = run_command(
            ["docker", "exec", container_name, "cat", f"/proc/{pid}/stat"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        closing_parenthesis = process_stat.rfind(")")
        stat_fields_after_command = process_stat[closing_parenthesis + 2 :].split()
        if closing_parenthesis < 0 or len(stat_fields_after_command) <= 19:
            raise ValueError(f"cannot parse process start time for pid {pid}")
        process_start_time_ticks = int(stat_fields_after_command[19])
        executable = run_command(
            ["docker", "exec", container_name, "readlink", f"/proc/{pid}/exe"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        executable_hash = run_command(
            ["docker", "exec", container_name, "sha256sum", executable],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()[0]
        working_directory = run_command(
            ["docker", "exec", container_name, "readlink", f"/proc/{pid}/cwd"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        marker_process = pid in marker_pids
        entrypoint_path = None
        entrypoint_sha256 = None
        entrypoint_token = next(
            (token for token in shlex.split(command)[1:] if token.endswith(".py")),
            None,
        )
        if entrypoint_token is not None:
            unresolved_entrypoint = (
                entrypoint_token
                if entrypoint_token.startswith("/")
                else posixpath.join(working_directory, entrypoint_token)
            )
            entrypoint_path = run_command(
                ["docker", "exec", container_name, "readlink", "-f", unresolved_entrypoint],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            entrypoint_sha256 = run_command(
                ["docker", "exec", container_name, "sha256sum", entrypoint_path],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()[0]
        environment = run_command(
            [
                "docker",
                "exec",
                container_name,
                "sh",
                "-c",
                'test -r /proc/"$1"/environ || exit 45; cat /proc/"$1"/environ',
                "sh",
                str(pid),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        workers.append(
            {
                "pid": pid,
                "parent_pid": parent_pid,
                "process_start_time_ticks": process_start_time_ticks,
                "command": command,
                "marker_process": marker_process,
                "executable_path": executable,
                "executable_sha256": executable_hash,
                "working_directory": working_directory,
                "entrypoint_path": entrypoint_path,
                "entrypoint_sha256": entrypoint_sha256,
                "environment_sha256": hashlib.sha256(environment.encode("utf-8")).hexdigest(),
            }
        )

    proc_captures: list[tuple[int, str]] = []
    open_paths: set[str] = set()
    for worker in workers:
        pid = worker["pid"]
        completed = run_command(
            [
                "docker",
                "exec",
                container_name,
                "sh",
                "-c",
                'test -d /proc/"$1" || exit 44; for fd in /proc/"$1"/fd/*; do readlink "$fd" || true; done',
                "sh",
                str(pid),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        proc_captures.append((pid, completed.stdout))
        for line in completed.stdout.splitlines():
            path = line.removesuffix(" (deleted)").strip()
            if path.startswith("/"):
                open_paths.add(path)

    git_commit = run_command(
        ["docker", "exec", container_name, "git", "-C", inference_repo_path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = run_command(
        [
            "docker",
            "exec",
            container_name,
            "git",
            "-C",
            inference_repo_path,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    proc_bytes = b"".join(
        str(pid).encode("ascii") + b"\0" + output.encode("utf-8") + b"\0"
        for pid, output in proc_captures
    )
    timestamp = captured_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return InferenceEnvironmentAudit.model_validate(
        {
            "schema_version": "acl6060_event_inference_environment_audit_v5",
            "run_id": run_id,
            "container_name": container_name,
            "container_id": container["Id"],
            "container_image_id": container["Image"].removeprefix("sha256:"),
            "container_read_only_rootfs": bool(
                container.get("HostConfig", {}).get("ReadonlyRootfs", False)
            ),
            "container_network_mode": container.get("HostConfig", {}).get("NetworkMode"),
            "capture_host": capture_host or socket.gethostname(),
            "captured_at_utc": timestamp,
            "capture_command": "docker_inspect_proc_tree_worker_discovery_git_v5",
            "capture_phase": capture_phase,
            "worker_command_match": worker_command_match,
            "worker_processes": workers,
            "docker_inspect_sha256": hashlib.sha256(inspect.stdout.encode("utf-8")).hexdigest(),
            "process_listing_sha256": hashlib.sha256(
                process_listing.stdout.encode("utf-8")
            ).hexdigest(),
            "proc_open_files_sha256": hashlib.sha256(proc_bytes).hexdigest(),
            "process_identity_tree_sha256": worker_process_identity_tree_sha256(workers),
            "inference_repo_path": inference_repo_path,
            "inference_git_commit": git_commit,
            "inference_git_status_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
            "forbidden_container_artifact_roots": normalized_container_roots,
            "forbidden_host_mount_source_roots": normalized_host_roots,
            "observed_mounts": mounts,
            "process_open_file_paths": sorted(open_paths),
            "forbidden_artifact_exposure_detected": False,
        }
    )
