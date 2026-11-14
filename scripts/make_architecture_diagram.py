"""Draws reports/architecture.png (run once: python scripts/make_architecture_diagram.py)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "architecture.png"
fig, ax = plt.subplots(figsize=(13, 6.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

def arrow(x1, y1, x2, y2, label=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#222"))
    if label: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7.5, color="#a00")

blue, green, orange, grey, pink = "#dbe9f6", "#dcf0dc", "#fde9cc", "#eeeeee", "#f8dada"
ax.text(6.5, 5.95, "Recommendation pipeline - orchestrated by Prefect flow 'recsys-pipeline'", ha="center", fontsize=11, weight="bold")

box(0.2, 3.9, 1.8, 1.0, "1. Ingestion\nMovieLens-100K /\nsynthetic fallback", blue)
box(2.5, 3.9, 1.8, 1.0, "2. Validation\nschema, dup, range,\nref. integrity", blue)
box(4.8, 3.9, 1.8, 1.0, "3. Preparation\nclean, impute,\nstandardise", blue)
box(7.1, 3.9, 1.8, 1.0, "4. Transformation\ntime split, user/item\nsummaries, matrix", blue)
box(9.4, 3.9, 1.8, 1.0, "5. Feature store\nuser / item /\ninteraction features", green)
for x in (2.0, 4.3, 6.6, 8.9): arrow(x, 4.4, x + 0.5, 4.4)

box(2.5, 2.3, 1.8, 0.9, "data/errors/\nrejected rows +\nerror_reason", pink)
arrow(3.4, 3.9, 3.4, 3.2, "invalid")
box(2.5, 0.6, 1.8, 0.9, "reports/\nvalidation_report", grey); arrow(3.4, 2.3, 3.4, 1.5)

box(9.4, 2.3, 1.8, 1.0, "6. Training\nPopularity baseline\nSVD collaborative filter", orange)
arrow(10.3, 3.9, 10.3, 3.3)
box(11.6, 2.3, 1.3, 1.0, "MLflow\ntracking:\nparams, metrics,\nmodels", orange)
arrow(11.2, 2.8, 11.6, 2.8)
box(7.1, 2.3, 1.8, 0.9, "train / test split\n(test = future\ninteractions)", grey)
arrow(8.0, 3.9, 8.0, 3.2)
box(9.4, 0.6, 1.8, 1.1, "7. Evaluation\nP/R/NDCG@K, coverage,\ndiversity, RMSE/MAE", orange)
arrow(10.3, 2.3, 10.3, 1.7)
arrow(8.9, 2.6, 9.5, 1.7, "test")
arrow(11.2, 1.6, 11.9, 2.3, "metrics")
box(11.6, 0.6, 1.3, 0.8, "reports/\nmodel_comparison", grey); arrow(11.2, 1.0, 11.6, 1.0)
plt.tight_layout(); plt.savefig(OUT, dpi=160); print("saved", OUT)
