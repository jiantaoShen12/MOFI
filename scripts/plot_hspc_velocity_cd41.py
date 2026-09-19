"""Draw the velocity-only HSPC z=0 CD41 check from its generated table."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
table_dir = ROOT / "results/figure_4_velocity_finetuned_paper/coarse_dense/sample/tables/tf/reconstruction_primary/primary/batch/a_itga2b_cd41"
out = ROOT / "results/figure_4_velocity_finetuned_paper/coarse_dense/sample/figures/tf/paper_primary/primary/batch/perturb_batch_curve__task-a-itga2b-cd41__z0_velocity_finetuned.png"
out.parent.mkdir(parents=True, exist_ok=True)
sim = pd.read_csv(table_dir / "sim_curve__z-0.csv")
real = pd.read_csv(table_dir / "real_curve.csv")
plt.rcParams["font.family"] = "Arial"
fig, ax = plt.subplots(figsize=(7.0, 5.0))
right = ax.twinx()
left_line, = ax.plot(sim["time"], sim["primary_value"], color="#377eb8", linewidth=2.2, label="sim ITGA2B (z=0)")
left_real = ax.scatter(real["time"], real["primary_value"], color="#1f4e79", marker="o", s=52, edgecolors="white", zorder=5, label="real ITGA2B")
right_line, = right.plot(sim["time"], sim["secondary_value"], color="#e66101", linestyle="--", linewidth=2.2, label="sim CD41 (z=0)")
right_real = right.scatter(real["time"], real["secondary_value"], color="#b2182b", marker="X", s=60, edgecolors="white", zorder=5, label="real CD41")
ax.set_xlabel("Time", fontweight="bold")
ax.set_ylabel("RNA value (ITGA2B)", color="#377eb8", fontweight="bold")
right.set_ylabel("Protein value (CD41)", color="#e66101", fontweight="bold")
ax.tick_params(axis="y", colors="#377eb8")
right.tick_params(axis="y", colors="#e66101")
ax.grid(True, axis="y", alpha=0.2)
ax.spines["top"].set_visible(False)
right.spines["top"].set_visible(False)
handles = [left_line, right_line, left_real, right_real]
ax.legend(handles, [h.get_label() for h in handles], frameon=False, loc="upper left", bbox_to_anchor=(1.12, 1.0))
fig.suptitle("ITGA2B→CD41 (velocity-only fine-tuned z=0)", y=1.02)
fig.tight_layout()
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
print(out)
