import subprocess
import time
import sys

kernel_ref = "aleixlopez/biohub-0-946-edge-feature-tta-tuned"
competition_name = "biohub-cell-tracking-during-development"
version = "2"

def check_status() -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_ref],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    output = result.stdout.strip()
    if "RUNNING" in output:
        return "RUNNING"
    elif "COMPLETE" in output:
        return "COMPLETE"
    elif "ERROR" in output:
        return "ERROR"
    else:
        # Fallback parsing just in case of output format changes
        if "KernelWorkerStatus.RUNNING" in output or "RUNNING" in result.stderr:
            return "RUNNING"
        elif "KernelWorkerStatus.COMPLETE" in output or "COMPLETE" in result.stderr:
            return "COMPLETE"
        elif "KernelWorkerStatus.ERROR" in output or "ERROR" in result.stderr:
            return "ERROR"
        return "UNKNOWN"

print(f"Monitoring Kaggle kernel: {kernel_ref} for version {version}...", flush=True)
retry_count = 0

while True:
    status = check_status()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}", flush=True)
    
    if status == "COMPLETE":
        print("Kernel completed successfully! Initiating submission to Kaggle...", flush=True)
        submit_cmd = [
            "kaggle", "competitions", "submit",
            "-c", competition_name,
            "-k", kernel_ref,
            "-f", "submission.csv",
            "-v", version,
            "-m", f"Biohub 0.946 Edge-Feature TTA Tuned Baseline Version {version}"
        ]
        submit_result = subprocess.run(
            submit_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
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
        print(f"Unknown status. Raw output check failed. Retry count: {retry_count}", flush=True)
        if retry_count > 10:
            print("Too many unknown status retries. Exiting.", flush=True)
            break
            
    time.sleep(120)  # Sleep for 2 minutes between checks
