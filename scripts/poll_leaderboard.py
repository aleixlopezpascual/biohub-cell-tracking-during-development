#!/usr/bin/env python3
"""Automated poller to monitor Kaggle submission evaluation queue and record scores."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import sys


def get_submissions() -> list[dict]:
    cmd = [
        "kaggle",
        "competitions",
        "submissions",
        "biohub-cell-tracking-during-development",
        "--format",
        "json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return []
    try:
        return json.loads(res.stdout)
    except Exception:
        return []


def main() -> None:
    print("=" * 78)
    print("BIOHUB LIVE SCORING WATCHER STARTED")
    print("Polling Kaggle submission queue every 60 seconds...")
    print("=" * 78, flush=True)

    previous_pending_refs = set()

    while True:
        subs = get_submissions()
        if not subs:
            time.sleep(60)
            continue

        current_pending = []
        just_scored = []

        for s in subs:
            ref = s.get("ref")
            status = str(s.get("status", ""))
            score_str = str(s.get("publicScore", "")).strip()

            if "PENDING" in status:
                current_pending.append(ref)
                previous_pending_refs.add(ref)
            elif "COMPLETE" in status and ref in previous_pending_refs and score_str:
                just_scored.append((ref, score_str, s.get("description", "")))
                previous_pending_refs.remove(ref)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if just_scored:
            print("\n" + "🎉" * 30, flush=True)
            print(f"[{timestamp}] NEW SUBMISSION SCORED ON KAGGLE!", flush=True)
            for ref, score, desc in just_scored:
                print(f"  Submission ID: {ref}", flush=True)
                print(f"  Public Score:  {score}", flush=True)
                print(f"  Description:   {desc}", flush=True)
            print("🎉" * 30 + "\n", flush=True)

            # Trigger portfolio verification audit
            subprocess.run([sys.executable, "scripts/verify_final_selections.py"])

        if current_pending:
            print(f"[{timestamp}] Active in scoring queue: {len(current_pending)} submissions ({current_pending})", flush=True)
        else:
            print(f"[{timestamp}] All submissions evaluated! No pending jobs.", flush=True)
            break

        time.sleep(60)


if __name__ == "__main__":
    main()
