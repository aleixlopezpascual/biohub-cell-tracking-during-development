"""Monitor script to watch the Consensus Ensemble kernel execution on Kaggle and auto-submit on completion."""

import subprocess
import time
import sys

KERNEL_REF = "aleixlopez/biohub-consensus-ensemble-sota"
COMPETITION_NAME = "biohub-cell-tracking-during-development"
VERSION = "2"
MESSAGE = "Biohub Consensus Ensemble SOTA (0.946 Edge-TTA + 0.953 Subvoxel Flow Harmonic Bipartite Consensus) Version 2"


def check_status() -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_REF],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout.strip()
    if "RUNNING" in output or "KernelWorkerStatus.RUNNING" in output:
        return "RUNNING"
    elif "COMPLETE" in output or "KernelWorkerStatus.COMPLETE" in output:
        return "COMPLETE"
    elif "ERROR" in output or "KernelWorkerStatus.ERROR" in output:
        return "ERROR"
    elif "QUEUED" in output or "KernelWorkerStatus.QUEUED" in output:
        return "QUEUED"
    return "UNKNOWN"


def main():
    print(f"Monitoring Kaggle kernel: {KERNEL_REF} for version {VERSION}...", flush=True)
    retry_count = 0

    while True:
        status = check_status()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}", flush=True)

        if status == "COMPLETE":
            print("Kernel completed successfully! Initiating submission to Kaggle...", flush=True)
            submit_cmd = [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                COMPETITION_NAME,
                "-k",
                KERNEL_REF,
                "-f",
                "submission.csv",
                "-v",
                VERSION,
                "-m",
                MESSAGE,
            ]
            submit_result = subprocess.run(
                submit_cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("Submission command executed. Output:", flush=True)
            print(submit_result.stdout, flush=True)
            if submit_result.stderr:
                print("Errors/Warnings:", flush=True)
                print(submit_result.stderr, flush=True)
            break

        elif status == "ERROR":
            print("Kernel run failed with ERROR status on Kaggle. Exiting monitor script.", flush=True)
            break

        elif status == "UNKNOWN":
            retry_count += 1
            print(f"Unknown status. Retry count: {retry_count}", flush=True)
            if retry_count > 10:
                print("Too many unknown status retries. Exiting.", flush=True)
                break

        time.sleep(120)  # Check every 2 minutes


if __name__ == "__main__":
    main()
