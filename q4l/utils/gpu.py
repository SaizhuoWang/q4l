import subprocess


def kill_gpu_processes():
    # Use nvidia-smi to get the list of GPU process IDs
    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits"
    result = subprocess.run(
        cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        print("Error executing nvidia-smi:", result.stderr)
        return

    # Extract process IDs
    pids = result.stdout.strip().split("\n")

    # Kill each process
    for pid in pids:
        if pid:
            try:
                subprocess.run(
                    ["kill", "-9", pid],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(f"Killed process {pid}")
            except Exception as e:
                print(f"Failed to kill process {pid}. Error: {e}")


if __name__ == "__main__":
    kill_gpu_processes()
