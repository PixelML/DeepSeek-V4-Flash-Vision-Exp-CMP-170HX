#!/usr/bin/env python3
"""Sample sanitized per-GPU telemetry (index, mem, power, temp, clocks) to CSV."""
import csv
import os
import subprocess
import time

FIELDS = ["index", "memory.used", "power.draw", "temperature.gpu",
          "clocks.sm", "utilization.gpu", "clocks_throttle_reasons.active"]


def main():
    path = os.environ.get("DSV4_TELEMETRY", "results/telemetry.csv")
    duration = float(os.environ.get("DSV4_TELEMETRY_SECS", "1200"))
    interval = float(os.environ.get("DSV4_TELEMETRY_INTERVAL", "5"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t_end = time.time() + duration
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + FIELDS)
        while time.time() < t_end:
            out = subprocess.run([
                "nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
                "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip()
            row = [round(time.time(), 1)]
            for line in out.splitlines():
                w.writerow(row + [v.strip().replace(" ", "") for v in line.split(",")])
            f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main()
