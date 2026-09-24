#!/usr/bin/env python3
"""Automatically push the 0.948 inference kernel as soon as GPU quota resets, monitor it, and submit."""

import subprocess
import time
import sys
from pathlib import Path

kernel_dir = Path("scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta")
kernel_ref = "aleixlopez/biohub-0-948-momentum-deepcenter-tta"
competition_name = "biohub-cell-tracking-during-development"

def try_push() -> bool:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting to push kernel {kernel_ref}...", flush=True)
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = (res.stdout + "\n" + res.stderr).strip()
    print("Push response:\n" + output, flush=True)
    if "successfully pushed" in output.lower():
        return True
    return False

def check_status() -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_ref],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if "RUNNING" in output:
        return "RUNNING"
    elif "COMPLETE" in output:
        return "COMPLETE"
    elif "ERROR" in output:
        return "ERROR"
    elif "QUEUED" in output:
        return "QUEUED"
    return "UNKNOWN"

def main():
    print("=" * 60, flush=True)
    print(f"Starting Auto-Push & Submit Watcher for {kernel_ref}", flush=True)
    print("=" * 60, flush=True)

    # Phase 1: Wait for GPU quota reset and push successfully
    while True:
        if try_push():
            print("🚀 Successfully pushed kernel to Kaggle! Moving to monitoring phase...", flush=True)
            break
        print("⏳ GPU quota still reached. Sleeping for 15 minutes before retrying...", flush=True)
        time.sleep(900)

    # Phase 2: Monitor until completion
    print("=" * 60, flush=True)
    print(f"Monitoring execution of {kernel_ref} on Kaggle...", flush=True)
    print("=" * 60, flush=True)
    
    retry_count = 0
    while True:
        status = check_status()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}", flush=True)

        if status == "COMPLETE":
            print("🎉 Kernel completed successfully! Initiating submission to Kaggle...", flush=True)
            submit_cmd = [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                competition_name,
                "-k",
                kernel_ref,
                "-f",
                "submission.csv",
                "-v",
                "1",
                "-m",
                "Biohub 0.948 Momentum DeepCenter TTA Inference",
            ]
            submit_result = subprocess.run(
                submit_cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("Submission executed! Output:", flush=True)
            print(submit_result.stdout, flush=True)
            if submit_result.stderr:
                print("Submission stderr:", submit_result.stderr, flush=True)
            break
        elif status == "ERROR":
            print("❌ Kernel run failed with ERROR status. Please check logs.", flush=True)
            break
        elif status == "UNKNOWN":
            retry_count += 1
            if retry_count > 15:
                print("Too many unknown status retries. Exiting.", flush=True)
                break

        time.sleep(120)  # Sleep for 2 minutes between checks

if __name__ == "__main__":
    main()
