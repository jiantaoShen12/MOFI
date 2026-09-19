import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# Core ODE/SDE parameters
C_A, H_A = 0.5, 0.3
C_B, H_B = 1.5, 1.0
C_C, H_C = 1.0, 10.0
d_A, d_B, d_C = 0.4, 0.4, 0.4
sigma_A, sigma_B, sigma_C = 0.025, 0.025, 0.01
S = 1.5
dt = 0.1
time_points = [0, 6,12, 18, 24]

# Cell-cell placeholders (kept for compatibility with the original structure)
threshold = 0.5
k = 2

# Hill that defines D dimension and repulsion on (A, B)
sigma_D=0
HILL_CENTER_A = 2.0
HILL_CENTER_B = 1.5
HILL_SIGMA = 0.15
# Use a narrower force kernel + force clipping so XY snapshots stay close to baseline.
HILL_REPULSION_STRENGTH = 1
sigma_hill_A=0.05
sigma_hill_B=0.05

BASE_PATH = str(Path(__file__).resolve().parents[1])
Path(BASE_PATH).mkdir(parents=True, exist_ok=True)
GENERATOR_CONTRACT = {
    "name": "figure2_simulation",
    "seed": 42,
    "dt": dt,
    "simulation_time_points": time_points,
    "saved_time_divisor": 8,
    "monge_map": {
        "x_center": HILL_CENTER_A,
        "y_center": HILL_CENTER_B,
        "sigma_hill_A": sigma_hill_A,
        "sigma_hill_B": sigma_hill_B,
        "sigma_D": sigma_D,
    },
    "columns": {
        "primary": ["x1", "x2"],
        "secondary_joint": ["x1", "x2", "x4"],
        "derived_secondary": "x4",
    },
}
(Path(BASE_PATH) / "figure2_generator_contract.json").write_text(
    json.dumps(GENERATOR_CONTRACT, indent=2),
    encoding="utf-8",
)
VIEW_AZIM_AB = 45   # camera from A+ and B+ direction
VIEW_ELEV_ABC = 30
VIEW_ELEV_ABD = 56  # higher view along D-axis


def compute_d_dimension(x, y):
    return np.exp(-((x - HILL_CENTER_A) ** 2/ (2 * sigma_hill_A) + (y - HILL_CENTER_B) ** 2/(2 * sigma_hill_B))) + sigma_D * np.random.normal()


def repulsion_force_from_hill(
    x,
    y,
    strength=HILL_REPULSION_STRENGTH,
):

    D_value = np.exp(-((x - HILL_CENTER_A) ** 2/ (2 * sigma_hill_A) + (y - HILL_CENTER_B) ** 2/(2 * sigma_hill_B)))
    repulsion_x =  HILL_REPULSION_STRENGTH * (1 / sigma_hill_A) * D_value * (x - HILL_CENTER_A)
    repulsion_y =  HILL_REPULSION_STRENGTH * (1 / sigma_hill_B) * D_value * (y - HILL_CENTER_B)
    return repulsion_x, repulsion_y


def rgba_to_plotly(color, alpha=0.85):
    r, g, b = (int(np.clip(v, 0, 1) * 255) for v in color[:3])
    return f"rgba({r}, {g}, {b}, {alpha})"


# Initial conditions
np.random.seed(42)
initial_cells_1 = np.random.normal(
    loc=[2.8, 0.2, 0],           # 均值
    scale=[0.1, 0.1, 0.1],      # 各方向标准差：[方向1, 方向2, 方向3]
    size=(200, 3)
)
initial_cells_2 = np.random.normal([0, 0, 2], 0.1, size=(200, 3))
initial_cells = np.vstack((initial_cells_1, initial_cells_2))
initial_cells = np.maximum(initial_cells, 0)
initial_d = compute_d_dimension(initial_cells[:, 0], initial_cells[:, 1])
initial_cells = np.column_stack([initial_cells, initial_d])


# Euler-Maruyama step
def euler_maruyama_step(A, B, C, interaction_A, interaction_B, interaction_C):
    repulsion_A, repulsion_B = repulsion_force_from_hill(A, B)

    dA = (
        (C_A * A ** 2 + S) / (1 + C_A * A ** 2 + H_B * B ** 2 + H_C * C ** 2 + S)
        - d_A * A
        + interaction_A
        + repulsion_A
    ) * dt + sigma_A * np.random.normal() * np.sqrt(dt)

    dB = (
        (C_B * B ** 2 + S) / (1 + H_A * A ** 2 + C_B * B ** 2 + H_C * C ** 2 + S)
        - d_B * B
        + interaction_B
        + repulsion_B
    ) * dt + sigma_B * np.random.normal() * np.sqrt(dt)

    dC = (
        (C_C * C ** 2) / (1 + C_C * C ** 2)
        - d_C * C
        + interaction_C
    ) * dt + sigma_C * np.random.normal() * np.sqrt(dt)

    A = np.maximum(A + dA, 0.0)
    B = np.maximum(B + dB, 0.0)
    C = np.maximum(C + dC, 0.0)
    D = compute_d_dimension(A, B)
    return A, B, C, D


# Simulation with dynamic interaction scaffold
def simulate_cells_with_division_and_interaction(initial_cells_input, time_points_input):
    cell_states = {float(t): [] for t in time_points_input}
    current_cells = initial_cells_input.copy()
    cell_states[0.0] = current_cells.copy()

    max_time = max(time_points_input)
    total_steps = int(round(max_time / dt))
    record_steps = {int(round(t / dt)): float(t) for t in time_points_input}

    for step in range(1, total_steps + 1):
        N = len(current_cells)
        next_cells = []

        for i in range(N):
            cell = current_cells[i]
            A, B, C = cell[:3]

            distances = np.linalg.norm(current_cells[:, :3] - cell[:3], axis=1)
            interacting_indices = [j for j in range(N) if distances[j] < threshold and j != i]
            _ = interacting_indices

            interaction_A = 0.0
            interaction_B = 0.0
            interaction_C = 0.0

            A, B, C, D = euler_maruyama_step(A, B, C, interaction_A, interaction_B, interaction_C)
            next_cells.append([A, B, C, D])

            p_div = 0.05 * (B ** 2 / (1 + B ** 2)) * dt
            if np.random.rand() < p_div:
                next_cells.append([A, B, C, D])

        current_cells = np.array(next_cells)

        if step in record_steps:
            cell_states[record_steps[step]] = current_cells.copy()

    return cell_states


# Run simulation
cell_states = simulate_cells_with_division_and_interaction(initial_cells, time_points)

# Print counts
print("各时间点的细胞数量：")
for t in time_points:
    num_cells = len(cell_states[float(t)])
    print(f"时间 t={t}，细胞数量：{num_cells}")

final_time = float(max(time_points))
total_cells = len(cell_states[final_time])
print(f"\n最终时间 (t={int(final_time)}) 的总细胞数：{total_cells}")


# Plot 1: ABC trajectory snapshots
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))

for t, color in zip(time_points, colors):
    data_t = cell_states[float(t)]
    if len(data_t) > 0:
        ax.scatter(
            data_t[:, 0],
            data_t[:, 1],
            data_t[:, 2],
            c=[color],
            s=15,
            alpha=0.7,
            label=f"t={t}",
        )

ax.set_xlabel("A")
ax.set_ylabel("B")
ax.set_zlabel("C")
ax.set_title("3-D Cell States Evolution (A, B, C)")
ax.view_init(elev=VIEW_ELEV_ABC, azim=VIEW_AZIM_AB)
ax.legend()
plt.savefig(f"{BASE_PATH}/simulation_gene_3d1.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Plot 2: ABD trajectory snapshots, with T-surface background (same style as 3dsimulation)
x_grid = np.linspace(0, 3.5, 250)
y_grid = np.linspace(0, 2.5, 250)
X, Y = np.meshgrid(x_grid, y_grid)
Z = compute_d_dimension(X, Y)

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.35)

for t, color in zip(time_points, colors):
    data_t = cell_states[float(t)]
    if len(data_t) > 0:
        ax2.scatter(
            data_t[:, 0],
            data_t[:, 1],
            data_t[:, 3],
            c=[color],
            s=12,
            alpha=0.75,
            label=f"t={t}",
        )

ax2.set_xlabel("A")
ax2.set_ylabel("B")
ax2.set_zlabel("D")
ax2.set_title("3-D Cell States Evolution (A, B, D) with T Surface")
ax2.view_init(elev=VIEW_ELEV_ABD, azim=VIEW_AZIM_AB)
ax2.legend()
plt.savefig(f"{BASE_PATH}/simulation_T1.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{BASE_PATH}/simulation_abd_3d1.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

if HAS_PLOTLY:
    fig_html = go.Figure()
    fig_html.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis",
            opacity=0.35,
            showscale=False,
            name="T surface",
        )
    )
    for t, color in zip(time_points, colors):
        data_t = cell_states[float(t)]
        if len(data_t) > 0:
            fig_html.add_trace(
                go.Scatter3d(
                    x=data_t[:, 0],
                    y=data_t[:, 1],
                    z=data_t[:, 3],
                    mode="markers",
                    marker=dict(size=3, color=rgba_to_plotly(color)),
                    name=f"t={t}",
                )
            )

    fig_html.update_layout(
        title="3-D Cell States Evolution (A, B, D) with T Surface",
        scene=dict(
            xaxis_title="A",
            yaxis_title="B",
            zaxis_title="D",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.9)),
        ),
        margin=dict(l=0, r=0, b=0, t=42),
    )
    html_out = f"{BASE_PATH}/simulation_abd_3d.html"
    fig_html.write_html(html_out, include_plotlyjs="cdn", full_html=True)
    print(f"交互式3D图已保存至 {html_out}")
else:
    print("未检测到 plotly，跳过 HTML 交互图导出。")


# Optional 2D AB snapshot plot
plt.figure(figsize=(8, 6))
for t, color in zip(time_points, colors):
    data_t = cell_states[float(t)]
    if len(data_t) > 0:
        plt.scatter(data_t[:, 0], data_t[:, 1], c=[color], alpha=0.7, label=f"t={t}")

plt.xlabel("A")
plt.ylabel("B")
plt.title("Cell States Evolution (A, B)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{BASE_PATH}/simulation_gene1.png", dpi=300)
plt.close()


# Save snapshots to CSV
# x1=A, x2=B, x3=C, x4=D
frames = []
for t in time_points:
    data_t = cell_states[float(t)]
    frames.append(
        pd.DataFrame(
            {
                "samples": np.full(data_t.shape[0], t / 8),
                "x1": data_t[:, 0],
                "x2": data_t[:, 1],
                "x3": data_t[:, 2],
                "x4": data_t[:, 3],
            }
        )
    )

final_df = pd.concat(frames, ignore_index=True)
csv_out = f"{BASE_PATH}/simulation_gene.csv"
final_df.to_csv(csv_out, index=False)
print(f"所有时间点的数据已保存至 {csv_out}")
