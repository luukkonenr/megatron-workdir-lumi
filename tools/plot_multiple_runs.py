import os
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import os 

PROJECT_ROOT_NAME = "megatron-workdir-lumi"
current = Path(__file__).resolve()
root = current

while root.name != PROJECT_ROOT_NAME and root.parent != root:
    root = root.parent
if root.name != PROJECT_ROOT_NAME:
    raise RuntimeError(f"Could not find project root '{PROJECT_ROOT_NAME}' starting from {current}")
sys.path.insert(0, str(root))

from tools.throughput import extract_values

def plot_by_key(key, val_dict, out_dir: Path, plot_by, time_in_minutes):
    plt.figure(figsize=(10, 6))
    annotation_positions = []  # track used y positions for annotations
    for k, vals in val_dict.items():
        path = k.name

        y = vals.get(key)
        if not isinstance(y, (list, np.ndarray)):
            continue
        n = len(y)
        if plot_by == 'iters':
            x = np.arange(0, n) * int(vals['log_interval'])
        elif plot_by == 'time':
            x = np.linspace(0, time_in_minutes, n)
        else:
            assert False, "Not a valid option"
        plt.plot(x, y, label=f'{path}-{key}', linewidth=2)
        last_x = x[-1]; last_y = y[-1]
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylabel(key, fontsize=12)
        plt.ylim(1,8)
        plt.yscale('log')
        plt.xlabel(plot_by)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        plt.tight_layout()

        # Annotation with de-overlap:
        base_factor = 0.05  # 5% upward
        factor = base_factor
        min_rel_dist = 0.04  # require at least 4% separation
        ytext = last_y * (1 + factor)

        # Increase factor until no overlap with previous annotation positions
        while any(abs(ytext - used) / used < min_rel_dist for used in annotation_positions):
            factor += base_factor
            ytext = last_y * (1 + factor)
        annotation_positions.append(ytext)
        plt.annotate(f'{last_y:.4f}', xy=(last_x, last_y),
                     xytext=(last_x, ytext),
                     fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        
    # Save figure instead of showing
    filename = f"{key.replace(' ', '_')}.png"
    out_path = out_dir / filename
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Plot metrics from multiple Megatron-LM log files.")
    p.add_argument("--logs", nargs="+", required=True, help="Paths to log files.")
    p.add_argument("--out-dir", required=True, help="Directory to store generated plots.")
    p.add_argument("--keys", nargs="+", default=["lm loss"], help="Metric keys to plot.")
    p.add_argument("--extra-keys", nargs="*", default=["validation_loss"], help="Additional keys to extract (not necessarily plotted).")
    p.add_argument("--plot-by", choices=["iters", "time"], help="Walltime")
    p.add_argument("--time-in-minutes", type=int, default=3600)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = {}
    for p in args.logs:
        path_obj = Path(p)
        if not path_obj.is_file():
            print(f"Skipping missing file: {p}", file=sys.stderr)
            continue
        # Include all requested keys so extraction collects needed data
        extract_keys = list(set(args.keys + args.extra_keys))
        vals = extract_values(p, False, *extract_keys)[0]
        logs[path_obj] = vals
    
    if not logs:
        print("No valid logs to process.", file=sys.stderr)
        sys.exit(1)

    for key in args.keys:
        plot_by_key(key, logs, out_dir, plot_by=args.plot_by, time_in_minutes=args.time_in_minutes)

if __name__ == "__main__":
    main()