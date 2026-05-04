import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction_dashboard(times, metrics):
    methods = list(times.keys())

    # Extract values
    precision = [metrics[m]['precision'] for m in methods]
    recall = [metrics[m]['recall'] for m in methods]
    iou = [metrics[m]['IoU'] for m in methods]
    dice = [metrics[m]['Dice'] for m in methods]
    t = [times[m] for m in methods]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------------------------
    # 1. Precision–Recall scatter
    # -------------------------
    ax = axes[2]
    for i, m in enumerate(methods):
        ax.scatter(recall[i], precision[i])
        ax.text(recall[i] + 0.002, precision[i], m)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # -------------------------
    # 2. IoU / Dice bar plot
    # -------------------------
    ax = axes[1]
    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width/2, iou, width, label="IoU")
    ax.bar(x + width/2, dice, width, label="Dice")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Quality (IoU / Dice)")
    ax.legend()

    # -------------------------
    # 3. Time vs IoU (trade-off)
    # -------------------------
    ax = axes[0]
    for i, m in enumerate(methods):
        ax.scatter(t[i], iou[i])
        ax.text(t[i], iou[i], m)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("IoU")
    ax.set_xscale("log")  # important due to large variation
    ax.set_title("Speed vs Quality")
    ax.grid(True)

    plt.tight_layout()
    plt.show()