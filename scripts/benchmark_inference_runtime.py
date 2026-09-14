#!/usr/bin/env python3
"""Time an end-to-end inference command and sample peak GPU memory."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-limit-hours", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("an inference command is required after --")
    if args.runtime_limit_hours <= 0:
        parser.error("--runtime-limit-hours must be positive")
    return args


def _gpu_memory_mib() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    values = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    process = subprocess.Popen(args.command)
    peak_gpu_memory_mib: int | None = None
    timed_out = False
    limit_seconds = args.runtime_limit_hours * 3600
    while process.poll() is None:
        memory = _gpu_memory_mib()
        if memory is not None:
            peak_gpu_memory_mib = max(peak_gpu_memory_mib or 0, memory)
        if time.monotonic() - started >= limit_seconds:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        try:
            process.wait(timeout=min(1.0, limit_seconds - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            pass
    returncode = process.wait()
    wall_seconds = time.monotonic() - started
    payload = {
        "command": args.command,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "runtime_hours": wall_seconds / 3600,
        "runtime_limit_hours": args.runtime_limit_hours,
        "within_runtime_limit": not timed_out and wall_seconds <= limit_seconds,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
