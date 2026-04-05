import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker
import matplotlib.pyplot as plt

LOG_FILE = "/logs/metric_log.csv"
OUTPUT_FILE = "/logs/error_distribution.png"
SLEEP_SECONDS = 10

print("[plot] Plot service started.")

while True:
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            if len(df) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df["absolute_error"], bins=20, color="steelblue", edgecolor="black", alpha=0.8)
                ax.set_title(f"Distribution of Absolute Errors (n={len(df)})", fontsize=14)
                ax.set_xlabel("Absolute Error", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                plt.savefig(OUTPUT_FILE)
                plt.close(fig)
                print(f"[plot] Updated {OUTPUT_FILE} with {len(df)} records.")
            else:
                print("[plot] No data yet in metric_log.csv, waiting...")
        else:
            print(f"[plot] File {LOG_FILE} not found, waiting...")
    except Exception as e:
        print(f"[plot] Error: {e}")

    time.sleep(SLEEP_SECONDS)
