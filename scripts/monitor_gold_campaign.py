import subprocess
import time
import sys

kernel_ref = "aleixlopez/biohub-gold-oof-runner"

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
        if "KernelWorkerStatus.RUNNING" in output or "RUNNING" in result.stderr:
            return "RUNNING"
        elif "KernelWorkerStatus.COMPLETE" in output or "COMPLETE" in result.stderr:
            return "COMPLETE"
        elif "KernelWorkerStatus.ERROR" in output or "ERROR" in result.stderr:
            return "ERROR"
        return "UNKNOWN"

def get_logs() -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "logs", kernel_ref],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.strip()

print(f"Monitoring Kaggle Gold Campaign kernel: {kernel_ref}...", flush=True)
retry_count = 0

while True:
    status = check_status()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}", flush=True)
    
    if status == "COMPLETE":
        print("\n🎉 Kernel completed successfully! Fetching final execution logs:\n", flush=True)
        print("="*80, flush=True)
        print(get_logs(), flush=True)
        print("="*80, flush=True)
        break
        
    elif status == "ERROR":
        print("\n❌ Kernel run failed with ERROR status. Fetching error logs:\n", flush=True)
        print("="*80, flush=True)
        print(get_logs(), flush=True)
        print("="*80, flush=True)
        break
        
    elif status == "UNKNOWN":
        retry_count += 1
        print(f"Unknown status. Raw output check failed. Retry count: {retry_count}", flush=True)
        if retry_count > 10:
            print("Too many unknown status retries. Exiting.", flush=True)
            break
            
    time.sleep(120)  # Sleep for 2 minutes between checks
