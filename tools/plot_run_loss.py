from throughput import extract_values

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
def main():
    filepath=sys.argv[1]
    vals = extract_values(filepath, False)[0]

    y = vals['loss']
    n = len(y)
    x = np.arange(0, n) * 10
    x = x * int(vals['batch_size']) * int(vals['seq_len'])

    # Create the plot
    plt.figure(figsize=(10, 6))  # Bigger, better figure size
    plt.plot(x, y, label='Loss', color='royalblue', linewidth=2)

    # Add grid, title, labels
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f"Loss Over Consumed Tokens\n({filepath})", fontsize=14)
    plt.xlabel("Tokens Consumed", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    # Optional: Show ticks in nicer font
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Optional: Add legend
    plt.legend()

    # Save plot
    output_filename = filepath.replace(".out", "_loss_plot.png")
    output_filename = Path(output_filename).name

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()


if __name__=="__main__":
    main()