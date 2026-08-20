import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_RAW = "#2a78d6"
SERIES_TREND = "#eb6834"

def loadValLoss(path="ValidationMetrics.txt"):
    rows = []
    with open(path) as f:
        for line in f:
            epoch = int(line.split("Epoch: ")[1].split("/")[0])
            batch = int(line.split("Batch: ")[1].split("/")[0])
            batchesPerEpoch = int(line.split("Batch: ")[1].split("/")[1].split(",")[0])
            valLoss = float(line.split("ValLoss: ")[1])
            step = epoch * batchesPerEpoch + batch
            rows.append((step, valLoss))
    rows.sort()
    steps = np.array([r[0] for r in rows], dtype=np.float64)
    losses = np.array([r[1] for r in rows], dtype=np.float64)
    return steps, losses

steps, losses = loadValLoss()

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(steps, losses, color=SERIES_RAW, linewidth=1.2, alpha=0.85, label="Validation loss")

slope, intercept = np.polyfit(steps, losses, 1)
trend = slope * steps + intercept
ax.plot(steps, trend, color=SERIES_TREND, linewidth=2, linestyle="--", label="Linear trend")

ax.set_title("Validation loss over training", color=PRIMARY_INK, fontsize=14, pad=12)
ax.set_xlabel("Training batches", color=SECONDARY_INK)
ax.set_ylabel("Validation loss (NLL)", color=SECONDARY_INK)

ax.grid(True, color=GRIDLINE, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRIDLINE)
ax.tick_params(colors=MUTED_INK)

legend = ax.legend(frameon=False, labelcolor=SECONDARY_INK)

plt.tight_layout()
os.makedirs("Assets", exist_ok=True)
outPath = "Assets/validation_loss.png"
plt.savefig(outPath, facecolor=SURFACE)
plt.close(fig)
print(f"Saved {len(steps)} points to {outPath}")
