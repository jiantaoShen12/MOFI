import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import os
import pickle
import sys
import umap
import torch
from typing import Optional, Tuple
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import rcParams
from .plot_ode_v3 import plot_ode_v3 as plot_ode_v3_core
from umap import UMAP

rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Compatibility for paper UMAP pickles serialized by older numba versions.
# This preserves the fitted estimator; it does not fit a replacement UMAP.
try:  # pragma: no cover - only exercised by legacy release assets
    import numba.core.types.scalars as _numba_scalars
    from pynndescent.pynndescent_ import NNDescent as _NNDescent

    sys.modules.setdefault("numba.core.types.old_scalars", _numba_scalars)
    if not getattr(_NNDescent.__setstate__, "_mofi_legacy_compat", False):
        _nn_setstate = _NNDescent.__setstate__

        def _legacy_nn_setstate(self, state):
            state = dict(state)
            state.setdefault("quantization", None)
            state.setdefault("parallel_batch_queries", False)
            if "_min_distance" not in state:
                graph = state.get("_search_graph")
                graph_data = getattr(graph, "data", None)
                state["_min_distance"] = float(np.min(graph_data)) if graph_data is not None and len(graph_data) else 0.0
            return _nn_setstate(self, state)

        _legacy_nn_setstate._mofi_legacy_compat = True
        _NNDescent.__setstate__ = _legacy_nn_setstate
except Exception:
    pass
def train_save_umap(
    data: np.ndarray,
    save_path: str,
    n_neighbors: int = 10,
    min_dist: float = 0.2,
    n_components: int = 2,
    random_state: int = 42,
    metric: str = 'euclidean'
) -> umap.UMAP:
    """
    Train UMAP model and save to file
    
    Parameters:
        data: Training data (n_samples, n_features)
        save_path: Path to save UMAP model (.pkl file)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        n_components: Number of dimensions to reduce to
        random_state: Random seed
        metric: Distance metric
        
    Returns:
        Fitted UMAP model
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Train UMAP
    print(f"[train_save_umap] Training UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric=metric
    )
    reducer.fit(data)
    
    # Save model
    with open(save_path, 'wb') as f:
        pickle.dump(reducer, f)
    print(f"[train_save_umap] UMAP model saved to {save_path}")
    
    return reducer

def load_umap(
    load_path: str,
    expected_n_components: int = 2
) -> umap.UMAP:
    """
    Load UMAP model from file
    
    Parameters:
        load_path: Path to load UMAP model (.pkl file)
        expected_n_components: Expected number of components (for validation)
        
    Returns:
        Loaded UMAP model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model dimensions don't match expected
    """
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"UMAP model not found at {load_path}")
    
    # Load model
    with open(load_path, 'rb') as f:
        reducer = pickle.load(f)
    print(f"[load_umap] UMAP model loaded from {load_path}")
    
    # Validate dimensions
    if hasattr(reducer, 'n_components') and reducer.n_components != expected_n_components:
        raise RuntimeError(
            f"Loaded UMAP has n_components={reducer.n_components}, "
            f"but expected {expected_n_components}"
        )
    
    return reducer

def get_or_train_umap(
    data: np.ndarray,
    save_path: str,
    n_neighbors: int = 10,
    min_dist: float = 0.2,
    n_components: int = 2,
    random_state: int = 42,
    force_retrain: bool = False
) -> umap.UMAP:
    """
    Load existing UMAP model or train a new one if not exists
    
    Parameters:
        data: Training data (only used if model doesn't exist)
        save_path: Path to UMAP model file
        n_neighbors: Number of neighbors (only used for training)
        min_dist: Minimum distance (only used for training)
        n_components: Number of components
        random_state: Random seed (only used for training)
        force_retrain: If True, retrain even if model exists

    Returns:
        UMAP model (loaded or newly trained)
    """
    if not force_retrain and os.path.isfile(save_path):
        try:
            return load_umap(save_path, expected_n_components=n_components)
        except Exception as e:
            print(f"[get_or_train_umap] Failed to load existing model: {e}")
            print("[get_or_train_umap] Retraining UMAP...")
    
    return train_save_umap(
        data=data,
        save_path=save_path,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )

def apply_umap_transform(
    reducer: umap.UMAP,
    data: np.ndarray
) -> np.ndarray:
    """
    Apply UMAP transformation to data
    
    Parameters:
        reducer: Fitted UMAP model
        data: Data to transform (can be multi-dimensional array)
        
    Returns:
        Transformed data with same shape except last dimension
    """
    original_shape = data.shape
    *lead_dims, feature_dim = original_shape
    
    # Flatten to 2D
    data_flat = data.reshape(-1, feature_dim)
    
    # Transform
    data_transformed = reducer.transform(data_flat)
    
    # Reshape back
    new_shape = (*lead_dims, reducer.n_components)
    return data_transformed.reshape(new_shape)


def fitted_umap_background(reducer: umap.UMAP, data: np.ndarray) -> np.ndarray:
    """Return the fitted paper embedding when it matches the reference cells.

    Archived manuscript reducers retain ``embedding_`` in the exact cell order
    used for fitting.  Reusing it avoids a second approximate NN transform of
    the reference cells (which can visibly distort the paper UMAP), while new
    simulated states and trajectories still go through ``transform``.
    """
    fitted = getattr(reducer, "embedding_", None)
    if fitted is not None:
        fitted = np.asarray(fitted, dtype=float)
        if fitted.ndim == 2 and fitted.shape == (len(data), 2):
            return fitted
    return np.asarray(reducer.transform(np.asarray(data, dtype=np.float32)), dtype=float)

# ==================== 使用示例 ====================

def example_usage_in_plot_function(adata, output_path, dim_reduction='umap'):
    """
    示例：在绘图函数中如何使用统一的UMAP接口
    """
    if dim_reduction == 'umap':
        # UMAP模型保存路径
        umap_path = os.path.join(output_path, 'umap_model.pkl')
        
        # 获取或训练UMAP（自动处理缓存）
        reducer = get_or_train_umap(
            data=adata.obsm['X_latent'],
            save_path=umap_path,
            n_neighbors=10,
            min_dist=0.2,
            n_components=2,
            random_state=42
        )
        
        # 转换数据
        X_2d = reducer.transform(adata.obsm['X_latent'])
        
        # 如果有其他数据需要转换（如轨迹数据）
        # traj_2d = apply_umap_transform(reducer, traj_array)
        
        return X_2d, reducer


# TODO: add dim_reduction
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

def plot_growth(adata, dim_reduction='umap', output_path=None):
    if 'growth_rate' not in adata.obsm.keys():
        raise ValueError("Growth rate not found in adata.obsm.")

    X_latent = np.asarray(adata.obsm['X_latent'], dtype=np.float32)

    # 1. 获取降维数据
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm.keys():
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        plot_data = adata.obsm['X_umap']
    elif dim_reduction == 'umap_self':
        # 优先复用 ode_trajectories_v2 的“联合降维”思路（X + traj + point）
        # 这样 growth 图的背景形状会尽量与 ode_trajectories_v2 一致。
        D = X_latent.shape[-1]
        plot_data = None
        base_dir = None
        if output_path is not None:
            base_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)

        if base_dir:
            ode_dir = os.path.join(base_dir, "ode_results")
            ode_traj_path = os.path.join(ode_dir, "ode_traj.npy")
            ode_point_path = os.path.join(ode_dir, "ode_point.npy")

            if plot_data is None and os.path.isfile(ode_traj_path) and os.path.isfile(ode_point_path) and D > 2:
                try:
                    traj_array = np.load(ode_traj_path)
                    point_array = np.load(ode_point_path)
                    if traj_array.shape[-1] == D and point_array.shape[-1] == D:
                        traj_reshaped = traj_array.reshape(-1, D)
                        point_reshaped = point_array.reshape(-1, D)
                        temp_adata = sc.AnnData(
                            X=np.concatenate([X_latent, traj_reshaped, point_reshaped], axis=0)
                        )
                        x_raw_end = len(X_latent)
                        sc.pp.neighbors(temp_adata, n_neighbors=40, random_state=42)
                        sc.tl.umap(temp_adata, random_state=42)
                        plot_data = temp_adata.obsm['X_umap'][:x_raw_end]
                except Exception as e:
                    print(f"[plot_growth] fallback from ode-style UMAP due to: {e}")

            if plot_data is None:
                if D == 2:
                    plot_data = X_latent
                else:
                    reducer_pkl = os.path.join(ode_dir, "dim_reducer_umap.pkl")
                    reducer = get_or_train_umap(
                        data=X_latent,
                        save_path=reducer_pkl,
                        n_neighbors=50,
                        min_dist=0.2,
                        n_components=2,
                        random_state=42
                    )
                    plot_data = reducer.transform(X_latent)
        else:
            if D == 2:
                plot_data = X_latent
            else:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=50,
                    min_dist=0.2,
                    random_state=42,
                )
                plot_data = reducer.fit_transform(X_latent)

    elif dim_reduction == 'PCA':
        plot_data = adata.obsm['X_pca'][:, :2]
    elif dim_reduction in ['none', None]:
        plot_data = X_latent[:, :2]
    else:
        raise ValueError(f"Invalid dim_reduction: {dim_reduction}")

    g_values = adata.obsm['growth_rate'].flatten()


    # 3. 动态计算阈值，增强对比度
    # 使用 2% 和 98% 分位数来避免极端值干扰颜色范围
    vmin = np.percentile(g_values, 2)
    vmax = np.percentile(g_values, 98)
    
    # 设置画布
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    
    # 4. 绘制底色背景（可选）
    # 先用极浅的灰色画出所有细胞的轮廓，增加空间感
    ax.scatter(plot_data[:, 0], plot_data[:, 1], c='#e0e0e0', s=4, alpha=0.1, rasterized=True)

    # 5. 绘制生长率点
    # 使用 'magma' 或 'Spectral_r'，这些颜色在白色背景下更好看
    scatter = ax.scatter(
        plot_data[:, 0], plot_data[:, 1], 
        c=g_values, 
        cmap='RdYlBu_r',
        s=6,              # 稍微减小点的大小
        alpha=0.8,        # 提高透明度让色彩更实
        edgecolors='none', 
        vmin=vmin, 
        vmax=vmax,
        rasterized=True   # 如果细胞很多，开启这个防止导出的PDF过大
    )

    # 6. 精简界面
    ax.set_axis_off() # 直接关闭所有坐标轴线和标签
    
    # 7. 优雅的 Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False) # 去掉 colorbar 的外框
    cbar.set_label('Predicted Growth Rate', fontsize=12, labelpad=10)
    
    # 强制让背景为纯白
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=False)
        print(f"[plot_growth] saved to -> {output_path}")
    
    plt.show()
def plot_growthv1(adata, dim_reduction='umap', output_path=None):
    if 'growth_rate' not in adata.obsm.keys():
        raise ValueError("Growth rate not found in adata.obsm. Please check if the growth term is set or the model is trained.")
    
    # Get dimensionality reduction coordinates for plotting
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm.keys():
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        plot_data = adata.obsm['X_umap']
    elif dim_reduction == 'umap_self':
        # 直接用 latent 矩阵传入 _fit_umap 计算
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=150,
            min_dist=0.9,
            random_state=42,
        )
        plot_data = reducer.fit_transform(np.asarray(adata.obsm['X_latent'], dtype=np.float32))
    elif dim_reduction == 'PCA':
        plot_data = adata.obsm['X_pca'][:, :2]
    elif dim_reduction in ['none', None]:
        plot_data = adata.obsm['X_latent'][:, :2]
    else:
        raise ValueError(f"Invalid dim_reduction: {dim_reduction}")

    # Key modification: Reverse the order of points (last point becomes first, first point becomes last)
    plot_data = plot_data[::-1]  # Reverse coordinate order
    g_values = adata.obsm['growth_rate'][::-1]  # Synchronously reverse color value order (ensure one-to-one correspondence)

    # Calculate color mapping range for growth rate
    g_min = g_values.min()
    g_max = g_values.max()

    if g_max <= g_min:
        vmin, vmax = -1, 1
    elif g_max < 0:
        vmin, vmax = g_max, 0
    else:
        vmin = max(0, g_min)
        vmax = np.percentile(g_values, 95)

    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = plt.cm.RdYlBu_r(norm(g_values))

    # Plot scatter plot (now plotted in reversed order)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(plot_data[:, 0], plot_data[:, 1], c=colors, alpha=0.3, marker='o', s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array(g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Predicted Growth Rate')

    # Format color bar ticks
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{(x):.2f}'))

    # Save plot
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', transparent=True)
        print(f"[plot_growth] saved to -> {output_path}")
    plt.show()


from CytoBridge.tl.analysis import compute_map_X_latent

def plot_growth_map(adata, tranmap, dim_reduction='umap', output_path=None,device="cuda", umap_model_path=None):
    """
    基于转换后的潜在空间（X_latent_map）绘制生长率分布图
    
    参数:
        adata: AnnData对象，包含'growth_rate'和'X_latent'
        tranmap: TransportMap实例，用于计算转换后的潜在空间
        dim_reduction: 降维方法（'umap'/'PCA'/'none'）
        output_path: 图像保存路径（None则不保存）
        umap_model_path: Optional prefit UMAP model used for mapped latent coordinates.
    """
    # 检查并计算转换后的潜在空间
    if 'X_latent_map' not in adata.obsm:
        adata = compute_map_X_latent(adata, tranmap,device)
    
    # 检查生长率数据
    if 'growth_rate' not in adata.obsm:
        raise ValueError("growth_rate not found in adata.obsm. Please check if the growth term is set.")
    
    X_latent_map = np.asarray(adata.obsm['X_latent_map'], dtype=np.float32)

    # 创建临时AnnData用于降维计算
    temp_adata = sc.AnnData(X=X_latent_map)
    
    # 获取降维坐标
    if dim_reduction == 'umap':
        if umap_model_path is not None and str(umap_model_path).strip():
            with open(str(umap_model_path), 'rb') as f:
                reducer = pickle.load(f)
            plot_data = reducer.transform(np.asarray(adata.obsm['X_latent_map'], dtype=np.float32))
        elif 'X_umap' not in temp_adata.obsm:
            sc.pp.neighbors(temp_adata)
            sc.tl.umap(temp_adata)
            plot_data = temp_adata.obsm['X_umap']
        else:
            plot_data = temp_adata.obsm['X_umap']
    elif dim_reduction == 'umap_self':
        # 优先复用 ode_trajectories_v2 的“联合降维”思路（X + traj + point）
        # 这样 growth 图的背景形状会尽量与 ode_trajectories_v2 一致。
        D = X_latent_map.shape[-1]
        plot_data = None
        base_dir = None
        if output_path is not None:
            base_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)

        if base_dir:
            ode_dir = os.path.join(base_dir, "ode_results")
            ode_x_plot_path = os.path.join(ode_dir, "ode_x_map_plot.npy")
            ode_traj_path = os.path.join(ode_dir, "ode_traj_map.npy")
            ode_point_path = os.path.join(ode_dir, "ode_point_map.npy")

            if os.path.isfile(ode_x_plot_path):
                try:
                    cached_x_plot = np.load(ode_x_plot_path)
                    if cached_x_plot.shape[0] == len(X_latent_map) and cached_x_plot.shape[1] >= 2:
                        plot_data = cached_x_plot[:, :2]
                        print(f"[plot_growth_map] loaded exact background coordinates from {ode_x_plot_path}")
                except Exception as e:
                    print(f"[plot_growth_map] failed to load cached ode background: {e}")

            if plot_data is None and os.path.isfile(ode_traj_path) and os.path.isfile(ode_point_path) and D > 2:
                try:
                    traj_array = np.load(ode_traj_path)
                    point_array = np.load(ode_point_path)
                    if traj_array.shape[-1] == D and point_array.shape[-1] == D:
                        traj_reshaped = traj_array.reshape(-1, D)
                        point_reshaped = point_array.reshape(-1, D)
                        temp_adata = sc.AnnData(
                            X=np.concatenate([X_latent_map, traj_reshaped, point_reshaped], axis=0)
                        )

                        x_raw_end = len(X_latent_map)
                        sc.pp.neighbors(temp_adata, n_neighbors=40, random_state=42)
                        sc.tl.umap(temp_adata, random_state=42)
                        plot_data = temp_adata.obsm['X_umap'][:x_raw_end]
                except Exception as e:
                    print(f"[plot_growth] fallback from ode-style UMAP due to: {e}")

            if plot_data is None:
                if D == 2:
                    plot_data = X_latent_map
                else:
                    reducer_pkl = os.path.join(ode_dir, "dim_reducer_umap.pkl")
                    reducer = get_or_train_umap(
                        data=X_latent_map,
                        save_path=reducer_pkl,
                        n_neighbors=50,
                        min_dist=0.2,
                        n_components=2,
                        random_state=42
                    )
                    plot_data = reducer.transform(X_latent_map)
        else:
            if D == 2:
                plot_data = X_latent_map
            else:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=50,
                    min_dist=0.2,
                    random_state=42,
                )
                plot_data = reducer.fit_transform(X_latent_map)
    elif dim_reduction == 'PCA':
        if 'X_pca' not in temp_adata.obsm:
            sc.pp.pca(temp_adata, n_comps=2)
        plot_data = temp_adata.obsm['X_pca'][:, :2]
    elif dim_reduction in ['none', None]:
        plot_data = X_latent_map[:, :2]
    else:
        raise ValueError(f"Invalid dim_reduction: {dim_reduction}")
    
    # 反转点顺序（与原始plot_growth保持一致）
    plot_data = plot_data[::-1]
    g_values = adata.obsm['growth_rate'][::-1]
    
    # 计算颜色映射范围
    g_min = g_values.min()
    g_max = g_values.max()
    
    if g_max <= g_min:
        vmin, vmax = -1, 1
    elif g_max < 0:
        vmin, vmax = g_max, 0
    else:
        vmin = max(0, g_min)
        vmax = np.percentile(g_values, 95)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = plt.cm.RdYlBu_r(norm(g_values))
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(plot_data[:, 0], plot_data[:, 1], c=colors, alpha=0.3, marker='o', s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array(g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Predicted Growth Rate')
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # 保存图像
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', transparent=True)
        print(f"[plot_growth_map] 已保存到 -> {output_path}")
    plt.show()
#%%

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from CytoBridge.tl.analysis import generate_ode_trajectories
from scipy.interpolate import make_interp_spline

def plot_ode(X, point_array, traj_array, save_dir):
    """
    Adapt to data containing NaNs: Stop plotting subsequent parts of the trajectory when NaN is encountered; filter out circle points corresponding to NaNs
    """
    num_timepoints = len(X)
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, num_timepoints))
    cmap = LinearSegmentedColormap.from_list("custom_viridis", colors)

    # Style settings
    sns.set_style("white")
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'axes.edgecolor': 'lightgrey',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.color': 'dimgrey',
        'ytick.color': 'dimgrey',
        'font.size': 12,
    })

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Point size formula
    size = 80
    point = np.sqrt(size)
    gap_size = (point + point * 0.1) ** 2
    bg_size = (np.sqrt(gap_size) + point * 0.3) ** 2

    # 1) Real observations (filter out NaNs in original data)
    for t, data in enumerate(X):
        # Keep only samples without NaN in the current time step's original data
        valid_data = data[~np.isnan(data).any(axis=1)]
        if len(valid_data) == 0:  # Skip this time step if no valid data
            continue
        ax.scatter(
            valid_data[:, 0], valid_data[:, 1],
            c=[colors[t]] * len(valid_data),
            marker='X', s=80, alpha=0.25,
            label=f"T{t}.0"
        )

    # 2) Trajectory (Key modification: Stop plotting subsequent parts of the trajectory when NaN is encountered)
    dense_T = 300  # Number of dense time points after interpolation
    num_trajectories = traj_array.shape[1]  # Total number of trajectories (number of particles)

    for j in range(num_trajectories):
        # Extract the complete trajectory of the j-th particle (time steps, 2D coordinates)
        traj = traj_array[:, j, :]
        # Find the position where the first NaN appears in the trajectory (all before are valid segments)
        # Check if each row contains NaN, return the index of the first True (return trajectory length if no NaN)
        first_nan_idx = np.argmax(np.isnan(traj).any(axis=1))
        # If there is no NaN, the valid segment is the entire trajectory; otherwise, take up to the first NaN
        if not np.isnan(traj).any(axis=1).any():
            valid_traj = traj
        else:
            valid_traj = traj[:first_nan_idx]  # Keep only valid parts before the first NaN
        
        # Skip this trajectory if the length of the valid segment is 0
        if len(valid_traj) < 2:  # At least 2 points are needed for interpolation and plotting
            continue
        
        # Interpolate and smooth the valid segment (use only valid data)
        t_old = np.arange(len(valid_traj))  # Original time step indices (0,1,...,L-1, where L is valid length)
        t_new = np.linspace(0, t_old[-1], dense_T)  # Densified time points
        
        # Interpolate x and y coordinates (k=3 for cubic spline, requires at least 4 points; use linear interpolation k=1 if insufficient)
        k_order = 3 if len(valid_traj) >= 4 else 1
        fx = make_interp_spline(t_old, valid_traj[:, 0], k=k_order)
        fy = make_interp_spline(t_old, valid_traj[:, 1], k=k_order)
        x_smooth = fx(t_new)
        y_smooth = fy(t_new)

        # Plot the valid segment of this trajectory
        ax.plot(x_smooth, y_smooth,
                color='red', alpha=0.75, linewidth=1.2, solid_capstyle='round')

    # 3) Generated points (Key modification: Filter out circle points corresponding to NaNs)
    # Extract generated points at all time steps and filter out points containing NaNs
    gen_points_list = []  # Store all valid generated points
    gen_colors_list = []  # Store colors corresponding to valid points

    for t in range(point_array.shape[0]):  # Iterate over each time step
        # Extract all generated points at the t-th time step (number of particles, 2)
        points_t = point_array[t, :, :]
        # Filter valid points without NaN at the current time step
        valid_mask_t = ~np.isnan(points_t).any(axis=1)
        valid_points_t = points_t[valid_mask_t]
        if len(valid_points_t) == 0:  # Skip this time step if no valid points
            continue
        # Colors corresponding to valid points (repeat the color of the current time step)
        valid_colors_t = np.tile(colors[t], (len(valid_points_t), 1))
        
        # Add to list
        gen_points_list.append(valid_points_t)
        gen_colors_list.append(valid_colors_t)

    # Plot three-layer circles only if there are valid generated points
    if gen_points_list:
        # Merge all valid generated points and their corresponding colors
        gen_points = np.concatenate(gen_points_list, axis=0)
        gen_colors = np.concatenate(gen_colors_list, axis=0)
        
        # Plot three-layer concentric circles (outer black background → middle white gap → inner colored points)
        ax.scatter(gen_points[:, 0], gen_points[:, 1],
                   c='black', s=bg_size, alpha=1, marker='o', linewidths=0)  # Outermost black background
        ax.scatter(gen_points[:, 0], gen_points[:, 1],
                   c='white', s=gap_size, alpha=1, marker='o', linewidths=0)  # Middle white gap
        ax.scatter(gen_points[:, 0], gen_points[:, 1],
                   c=gen_colors, s=size, alpha=0.7, marker='o', linewidths=0)  # Innermost colored points

    # 4) Legend (Handle duplicate labels that may result from NaN filtering)
    # Main legend (ground truth, predicted points, trajectories)
    legend_main = [
        Line2D([0], [0], marker='X', color='w', label='Ground Truth',
               markerfacecolor=colors[0], markersize=10, alpha=0.3),
        Line2D([0], [0], marker='o', color='w', label='Predicted',
               markerfacecolor=colors[-1], markersize=10),
        Line2D([0], [0], color='red', lw=2, label='Trajectory')
    ]
    ax.legend(handles=legend_main, loc='upper right', frameon=True)

    # Time step legend (only display time steps with valid data)
    legend_t = []
    for t in range(num_timepoints):
        # Check if the current time step has valid original data (avoid displaying labels for time steps with no data)
        data_t = X[t]
        if len(data_t[~np.isnan(data_t).any(axis=1)]) > 0:
            legend_t.append(
                Line2D([0], [0], marker='X', color='w',
                       label=f'T{t}.0', markerfacecolor=colors[t], markersize=10, alpha=0.6)
            )
    if legend_t:  # Add time step legend only if there are valid time steps
        ax.legend(handles=legend_t, loc='upper left', title='Time Points', frameon=True)
        ax.add_artist(ax.get_legend())  # Keep the main legend

    ax.set_xlabel("Gene $X_1$")
    ax.set_ylabel("Gene $X_2$")
    ax.set_title("Real vs Generated Samples & Trajectories")

    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight')  # bbox_inches prevents legend from being cut off
    plt.show()
    #plt.close()
# ---------- V3 V3 V3 V3 V3 V3 V3 V3 V3 V3                     V3 V3 V3 V3 V3 V3 V3 V3 V3 V3----------
from matplotlib.collections import LineCollection


# def _to_label(value):
#     if value is None:
#         return "NA"
#     try:
#         if np.isnan(value):
#             return "NA"
#     except Exception:
#         pass
#     return str(value)


# def _soften_color(rgb, saturation_keep=0.78, white_mix=0.25):
#     desat_rgb = sns.desaturate(rgb, saturation_keep)
#     return tuple((1.0 - white_mix) * float(c) + white_mix for c in desat_rgb[:3])


# def _build_traj_cmap(traj_cmap):
#     name = str(traj_cmap).lower()
#     if name in {"sunset", "nature_v4"}:
#         base = LinearSegmentedColormap.from_list(
#             "nature_v3",
            
#             ["#57E020","#FFC829", "#CC4A77", "#2F2F85"]
#         )
#     elif name in {"nature_v3"}:
#         base = LinearSegmentedColormap.from_list(
#             "nature_v4",
#             ["#FFC829", "#CC4A77", "#2F2F85"]
#         )
#     elif name in {"flare", "mako", "rocket", "crest"}:
#         base = sns.color_palette(name, as_cmap=True)
#     else:
#         base = plt.get_cmap(name)
#     return LinearSegmentedColormap.from_list(
#         f"{name}_trimmed_v3",
#         base(np.linspace(0.08, 0.95, 256))
#     )


# def _build_bg_palette(labels_plot, bg_palette, max_legend_items=10):
#     unique_labels, counts = np.unique(labels_plot, return_counts=True)
#     order = unique_labels[np.argsort(counts)[::-1]]
#     if len(order) > max_legend_items:
#         keep_n = max(max_legend_items - 1, 1)
#         kept = set(order[:keep_n])
#         labels_plot = np.array(
#             [lab if lab in kept else "Other" for lab in labels_plot],
#             dtype=object
#         )
#         legend_labels = list(order[:keep_n]) + ["Other"]
#     else:
#         legend_labels = list(order)

#     palette = sns.color_palette(bg_palette, n_colors=len(legend_labels))
#     color_map = {
#         lab: _soften_color(palette[i], saturation_keep=0.80, white_mix=0.24)
#         for i, lab in enumerate(legend_labels)
#     }
#     colors = np.array([color_map[lab] for lab in labels_plot], dtype=float)
#     handles = [
#         Line2D(
#             [0], [0],
#             marker="o",
#             linestyle="None",
#             markerfacecolor=color_map[lab],
#             markeredgecolor=color_map[lab],
#             markeredgewidth=0.0,
#             markersize=5.2,
#             label=lab,
#         )
#         for lab in legend_labels
#     ]
#     return colors, handles


# def _draw_background(ax, bg_points, bg_labels, bg_palette, background_mode):
#     if bg_points.size == 0:
#         return None

#     if bg_labels is None:
#         mode = str(background_mode).lower()
#         if mode == "auto":
#             mode = "hexbin" if bg_points.shape[0] > 35000 else "scatter"

#         if mode == "hexbin":
#             ax.hexbin(
#                 bg_points[:, 0],
#                 bg_points[:, 1],
#                 gridsize=115,
#                 cmap="Greys",
#                 mincnt=1,
#                 linewidths=0,
#                 alpha=0.18,
#                 zorder=1,
#             )
#         elif mode == "contour":
#             ax.tricontourf(
#                 bg_points[:, 0],
#                 bg_points[:, 1],
#                 np.ones(bg_points.shape[0], dtype=float),
#                 levels=6,
#                 cmap="Greys",
#                 alpha=0.12,
#                 zorder=1,
#             )
#         else:
#             ax.scatter(
#                 bg_points[:, 0],
#                 bg_points[:, 1],
#                 s=4.0,
#                 c="#9DA3AD",
#                 marker="o",
#                 alpha=0.10,
#                 linewidths=0,
#                 rasterized=True,
#                 zorder=1,
#             )
#         return None

#     labels_plot = np.array([_to_label(v) for v in bg_labels], dtype=object)
#     bg_colors, legend_handles = _build_bg_palette(labels_plot, bg_palette, max_legend_items=10)
#     ax.scatter(
#         bg_points[:, 0],
#         bg_points[:, 1],
#         s=6.5,
#         c=bg_colors,
#         marker="o",
#         alpha=0.18,
#         linewidths=0,
#         rasterized=True,
#         zorder=1,
#     )
#     return legend_handles


# def _draw_endpoints(ax, traj, traj_times, start_idx, end_idx, norm, cmap):
#     start_pt = traj[start_idx]
#     end_pt = traj[end_idx]
#     start_color = cmap(norm(float(traj_times[start_idx])))
#     end_color = cmap(norm(float(traj_times[end_idx])))

#     # ax.scatter(
#     #     start_pt[0], start_pt[1],
#     #     s=45,
#     #     facecolors="none",
#     #     edgecolors="black",
#     #     linewidths=0.65,
#     #     zorder=4.2,
#     # )
#     ax.scatter(
#         start_pt[0], start_pt[1],
#         s=28,
#         facecolors=[start_color],  # 实心
#         edgecolors=[start_color],
#         linewidths=1.00,
#         zorder=4.3,
#     )

#     ax.scatter(
#         end_pt[0], end_pt[1],
#         s=58,
#         c=[end_color],
#         edgecolors="black",
#         linewidths=0.25,
#         zorder=4.4,
#     )
#     ax.scatter(
#         end_pt[0], end_pt[1],
#         s=92,
#         facecolors="none",
#         edgecolors=[(*end_color[:3], 0.30)],
#         linewidths=0.80,
#         zorder=4.1,
#     )


# def plot_ode_v3(
#     X,
#     traj_array,
#     save_path,
#     traj_times,
#     time_ticks=None,
#     bg_obs_values=None,
#     bg_obs_name=None,
#     bg_palette="Set2",
#     traj_cmap="sunset",
#     background_mode="auto",
#     show_axes=True,
#     dpi=300,
# ):
#     """
#     Nature-style trajectory rendering with smooth gradient flow and layered background.
#     """
#     if traj_array.ndim != 3 or traj_array.shape[-1] != 2:
#         raise ValueError(f"traj_array must have shape (T, M, 2), got {traj_array.shape}")

#     traj_times = np.asarray(traj_times, dtype=float)
#     if traj_times.shape[0] != traj_array.shape[0]:
#         raise ValueError("traj_times length must match traj_array first dimension")

#     sns.set_style("white")
#     plt.rcParams.update({
#         "font.family": "sans-serif",
#         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
#         "axes.grid": False,
#         "axes.linewidth": 0.72,
#         "axes.labelsize": 10,
#         "xtick.labelsize": 9,
#         "ytick.labelsize": 9,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#     })

#     has_bg_obs = bg_obs_values is not None
#     if has_bg_obs:
#         if len(bg_obs_values) != len(X):
#             raise ValueError("bg_obs_values must have same length as X")
#         fig = plt.figure(figsize=(7.8, 6.5), dpi=dpi)
#         gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.34], wspace=0.03)
#     else:
#         fig = plt.figure(figsize=(6.9, 6.5), dpi=dpi)
#         gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.13], wspace=0.03)

#     ax = fig.add_subplot(gs[0, 0])
#     side_ax = fig.add_subplot(gs[0, 1])
#     side_ax.set_axis_off()

#     bg_points_list = []
#     bg_labels_list = []
#     for idx, data_t in enumerate(X):
#         if data_t is None or len(data_t) == 0:
#             continue
#         valid_mask_t = ~np.isnan(data_t).any(axis=1)
#         valid_t = data_t[valid_mask_t]
#         if len(valid_t) == 0:
#             continue
#         bg_points_list.append(valid_t[:, :2])
#         if has_bg_obs:
#             labels_t = np.asarray(bg_obs_values[idx])
#             if labels_t.shape[0] != data_t.shape[0]:
#                 raise ValueError("Each bg_obs_values[i] must align with X[i] rows")
#             bg_labels_list.append(labels_t[valid_mask_t])

#     bg_points = np.vstack(bg_points_list) if bg_points_list else np.empty((0, 2), dtype=float)
#     bg_labels = np.concatenate(bg_labels_list, axis=0) if has_bg_obs and bg_labels_list else None
#     legend_handles = _draw_background(
#         ax=ax,
#         bg_points=bg_points,
#         bg_labels=bg_labels,
#         bg_palette=bg_palette,
#         background_mode=background_mode,
#     )

#     cmap = _build_traj_cmap(traj_cmap)
#     norm = Normalize(vmin=float(np.nanmin(traj_times)), vmax=float(np.nanmax(traj_times)))

#     n_trajectories = traj_array.shape[1]
#     arrow_budget = 18
#     arrows_drawn = 0
#     for j in range(n_trajectories):
#         traj = traj_array[:, j, :]
#         valid_mask = ~np.isnan(traj).any(axis=1)
#         if valid_mask.sum() == 0:
#             continue

#         valid_idx = np.where(valid_mask)[0]
#         split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
#         idx_chunks = np.split(valid_idx, split_points)

#         for chunk in idx_chunks:
#             if chunk.size < 2:
#                 continue
#             pts = traj[chunk]
#             t_chunk = traj_times[chunk]
#             segments = np.stack([pts[:-1], pts[1:]], axis=1)
#             seg_t = 0.5 * (t_chunk[:-1] + t_chunk[1:])
#             seg_alpha = np.linspace(0.45, 1.0, len(seg_t))

#             base_colors = cmap(norm(seg_t))
#             glow_colors = base_colors.copy()
#             glow_colors[:, 3] = 0.11 * seg_alpha
#             main_colors = base_colors.copy()
#             main_colors[:, 3] = 0.78 * seg_alpha

#             lc_glow = LineCollection(
#                 segments,
#                 colors=glow_colors,
#                 linewidths=3.9,
#                 capstyle="round",
#                 joinstyle="round",
#                 zorder=2,
#             )
#             lc_main = LineCollection(
#                 segments,
#                 colors=main_colors,
#                 linewidths=1.4,
#                 capstyle="round",
#                 joinstyle="round",
#                 zorder=3,
#             )
#             ax.add_collection(lc_glow)
#             ax.add_collection(lc_main)

#             if arrows_drawn < arrow_budget and chunk.size >= 8:
#                 mid = int(0.50 * (chunk.size - 1))
#                 p0 = pts[max(mid - 1, 0)]
#                 p1 = pts[min(mid + 1, chunk.size - 1)]
#                 arrow_color = cmap(norm(float(t_chunk[mid])))
#                 ax.annotate(
#                     "",
#                     xy=p1,
#                     xytext=p0,
#                     arrowprops=dict(
#                         arrowstyle="-|>",
#                         color=arrow_color,
#                         lw=0.0,
#                         shrinkA=0,
#                         shrinkB=0,
#                         mutation_scale=8.0,
#                         alpha=0.85,
#                     ),
#                     zorder=3.4,
#                 )
#                 arrows_drawn += 1

#         start_idx = int(valid_idx[0])
#         end_idx = int(valid_idx[-1])
#         _draw_endpoints(ax, traj, traj_times, start_idx, end_idx, norm, cmap)

#     sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     if has_bg_obs:
#         cax = side_ax.inset_axes([0.43, 0.54, 0.22, 0.40])
#     else:
#         cax = side_ax.inset_axes([0.42, 0.12, 0.22, 0.78])
#     cbar = fig.colorbar(sm, cax=cax)
#     cbar.set_label("Time Point")
#     cbar.outline.set_linewidth(0.6)
#     cbar.ax.tick_params(length=2.2, width=0.6, pad=1.5)
#     if time_ticks is not None:
#         ticks = np.unique(np.asarray(time_ticks, dtype=float))
#         if ticks.size > 0:
#             cbar.set_ticks(ticks)

#     if legend_handles:
#         side_ax.legend(
#             handles=legend_handles,
#             title=(bg_obs_name if bg_obs_name else "Background"),
#             title_fontsize=8,
#             fontsize=7.5,
#             loc="lower left",
#             bbox_to_anchor=(0.00, 0.02),
#             frameon=False,
#             borderaxespad=0.0,
#             handletextpad=0.3,
#             labelspacing=0.24,
#         )

#     ax.set_aspect("equal", adjustable="box")
#     if show_axes:
#         ax.set_xlabel("Embedding 1")
#         ax.set_ylabel("Embedding 2")
#         ax.tick_params(length=2.8, width=0.7)
#         sns.despine(ax=ax, top=True, right=True)
#     else:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#     ax.grid(False)

#     plt.tight_layout()
#     save_parent = os.path.dirname(save_path)
#     if save_parent:
#         os.makedirs(save_parent, exist_ok=True)
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close(fig)

# ---------- Parent function: responsible for dimensionality reduction + plotting ----------



# ==================== 修改后的 plot_ode_trajectories ====================




# ==================== 修改后的 plot_ode_trajectories ====================
def plot_ode_trajectories(
    adata,
    model,
    output_path='figures',
    save_base_name='ode_trajectories.pdf',
    n_trajectories=50,
    n_bins=100,
    dim_reduction='umap',
    device="cuda",
    split_true=False
):
    model.to(device)
    model.eval()
    output_path = os.path.join(output_path, "ode_results")
    os.makedirs(output_path, exist_ok=True)

    point_array, traj_array = generate_ode_trajectories(
        model, adata,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
        exp_dir = output_path,          
        split_true=split_true
    )

    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())
    X_raw = adata.obsm['X_latent']
    D = traj_array.shape[-1]

    # 2. Reduce dimension to 2-D
    if D == 2:
        traj_2d, point_2d = traj_array, point_array
        X_2d = X_raw
    else:
        if dim_reduction == 'none':
            traj_2d  = traj_array[..., :2]
            point_2d = point_array[..., :2]
            X_2d     = X_raw[..., :2]
        elif dim_reduction in ('pca', 'umap'):
            reducer_pkl = os.path.join(output_path, f"dim_reducer_{dim_reduction}.pkl")
            
            if dim_reduction == 'pca':
                
                if os.path.isfile(reducer_pkl):
                    with open(reducer_pkl, 'rb') as f:
                        reducer = pickle.load(f)
                    print(f"[dim-reduction] loaded existing PCA from {reducer_pkl}")
                else:
                    print("[dim-reduction] training PCA...")
                    reducer = PCA(n_components=2, random_state=0)
                    reducer.fit(X_raw)
                    with open(reducer_pkl, 'wb') as f:
                        pickle.dump(reducer, f)
                    print(f"[dim-reduction] PCA saved to {reducer_pkl}")
                
                X_2d = fitted_umap_background(reducer, X_raw)
                traj_2d = apply_umap_transform(reducer, traj_array)
                point_2d = apply_umap_transform(reducer, point_array)
                
            elif dim_reduction == 'umap':
                # 使用统一的UMAP接口
                reducer = get_or_train_umap(
                    data=X_raw,
                    save_path=reducer_pkl,
                    n_neighbors=10,
                    min_dist=0.2,
                    n_components=2,
                    random_state=42
                )
                
                X_2d = reducer.transform(X_raw)
                traj_2d = apply_umap_transform(reducer, traj_array)
                point_2d = apply_umap_transform(reducer, point_array)
        elif dim_reduction == 'umap_adata':
            # 获取traj_array和point_array的形状信息
            T_bins, M, _ = traj_array.shape
            T, _, _ = point_array.shape
            
            # 调整traj_array和point_array的形状为二维
            traj_reshaped = traj_array.reshape(-1, D)  # (T_bins*M, D)
            point_reshaped = point_array.reshape(-1, D)  # (T*M, D)
            
            # 创建临时adata，包含所有需要降维的数据
            temp_adata = sc.AnnData(
                X=np.concatenate([X_raw, traj_reshaped, point_reshaped], axis=0)
            )
            # 存储原始数据的索引边界
            x_raw_end = len(X_raw)
            traj_end = x_raw_end + len(traj_reshaped)
            
            # 使用X_latent计算邻居并进行UMAP降维
            # temp_adata.obsm['X_latent'] = temp_adata.X.copy()  # 将数据存入X_latent
            # sc.pp.neighbors(temp_adata, use_rep='X_latent')
            # sc.tl.umap(temp_adata)
            
            sc.pp.neighbors(temp_adata)
            sc.tl.umap(temp_adata)
            
            # 提取降维后的结果
            X_2d = temp_adata.obsm['X_umap'][:x_raw_end]  # (batchsize, 2)
            traj_2d_unshape = temp_adata.obsm['X_umap'][x_raw_end:traj_end]  # (T_bins*M, 2)
            point_2d_unshape = temp_adata.obsm['X_umap'][traj_end:]  # (T*M, 2)
            
            # 恢复原始形状
            traj_2d = traj_2d_unshape.reshape(T_bins, M, 2)  # (T_bins, M, 2)
            point_2d = point_2d_unshape.reshape(T, M, 2)  # (T, M, 2)
        else:
            raise ValueError("dim_reduction must be 'none', 'pca', 'umap' or 'umap_adata'")

    # Slice by time
    X = [X_2d[adata.obs[samples_key] == t] for t in unique_times]

    # 3. Save & plot
    np.save(os.path.join(output_path, "ode_traj_2d.npy"),  traj_2d)
    np.save(os.path.join(output_path, "ode_point_2d.npy"), point_2d)
    print(f"[generate_ode_trajectories] trajectories saved to {output_path}")

    output_path = os.path.join(output_path, save_base_name)
    plot_ode(X=X,
             point_array=point_2d,
             traj_array=traj_2d,
             save_dir=output_path)

    print(f"[ode_trajectories] done -> {output_path}")



def plot_ode_trajectories_v2(
    adata,
    model,
    output_path='figures',
    save_base_name='ode_trajectories_v2.pdf',
    n_trajectories=50,
    n_bins=100,
    dim_reduction='umap',
    bg_obs_key='time_point_processed',
    bg_palette="Set2",
    traj_cmap="sunset1",
    background_mode="auto",
    show_axes=True,
    device="cuda",
    split_true=False,
    sampling_mode="random",
    init_time=None,
    init_cell_type=None,
    init_indices=None,
    cell_type_key="cell_type",
    sampling_seed=None,
    strict_index_init_time=True,
    perturbation_cfg=None,
    perturbation_reference_adata=None,
):
    model.to(device)
    model.eval()
    output_path = os.path.join(output_path, "ode_results")
    os.makedirs(output_path, exist_ok=True)

    point_array, traj_array, sampling_meta = generate_ode_trajectories(
        model, adata,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
        exp_dir=output_path,
        split_true=split_true,
        sampling_mode=sampling_mode,
        init_time=init_time,
        init_cell_type=init_cell_type,
        init_indices=init_indices,
        cell_type_key=cell_type_key,
        sampling_seed=sampling_seed,
        strict_index_init_time=strict_index_init_time,
        perturbation_cfg=perturbation_cfg,
        perturbation_reference_adata=perturbation_reference_adata,
        return_metadata=True,
    )

    samples_key = 'time_point_processed'
    unique_times = np.sort(np.asarray(adata.obs[samples_key].unique(), dtype=float))
    sim_time_ticks = np.asarray(sampling_meta.get("time_points", unique_times), dtype=float)
    if sim_time_ticks.size == 0:
        sim_time_ticks = unique_times
    X_raw = adata.obsm['X_latent']
    D = traj_array.shape[-1]
    print(X_raw.shape)

    if D == 2:
        traj_2d, point_2d = traj_array, point_array
        X_2d = X_raw
    else:
        if dim_reduction == 'none':
            traj_2d = traj_array[..., :2]
            point_2d = point_array[..., :2]
            X_2d = X_raw[..., :2]
        elif dim_reduction in ('pca', 'umap'):
            reducer_pkl = os.path.join(output_path, f"dim_reducer_{dim_reduction}.pkl")

            if dim_reduction == 'pca':
                
                if os.path.isfile(reducer_pkl):
                    with open(reducer_pkl, 'rb') as f:
                        reducer = pickle.load(f)
                    print(f"[dim-reduction] loaded existing PCA from {reducer_pkl}")
                else:
                    print("[dim-reduction] training PCA...")
                    reducer = PCA(n_components=2, random_state=0)
                    reducer.fit(X_raw)
                    with open(reducer_pkl, 'wb') as f:
                        pickle.dump(reducer, f)
                    print(f"[dim-reduction] PCA saved to {reducer_pkl}")

                X_2d = reducer.transform(X_raw)
                traj_2d = apply_umap_transform(reducer, traj_array)
                point_2d = apply_umap_transform(reducer, point_array)

            elif dim_reduction == 'umap':
                reducer = get_or_train_umap(
                    data=X_raw,
                    save_path=reducer_pkl,
                    n_neighbors=50,
                    min_dist=0.2,
                    n_components=2,
                    random_state=42
                )

                X_2d = reducer.transform(X_raw)
                traj_2d = apply_umap_transform(reducer, traj_array)
                point_2d = apply_umap_transform(reducer, point_array)
        elif dim_reduction == 'umap_adata':
            T_bins, M, _ = traj_array.shape
            T, _, _ = point_array.shape

            traj_reshaped = traj_array.reshape(-1, D)
            point_reshaped = point_array.reshape(-1, D)

            temp_adata = sc.AnnData(
                X=np.concatenate([X_raw, traj_reshaped, point_reshaped], axis=0)
            )
            x_raw_end = len(X_raw)
            traj_end = x_raw_end + len(traj_reshaped)

            sc.pp.neighbors(temp_adata, n_neighbors=40, random_state=42)
            sc.tl.umap(temp_adata)
            X_2d = temp_adata.obsm['X_umap'][:x_raw_end]
            traj_2d_unshape = temp_adata.obsm['X_umap'][x_raw_end:traj_end]
            point_2d_unshape = temp_adata.obsm['X_umap'][traj_end:]

            traj_2d = traj_2d_unshape.reshape(T_bins, M, 2)
            point_2d = point_2d_unshape.reshape(T, M, 2)
        else:
            raise ValueError("dim_reduction must be 'none', 'pca', 'umap' or 'umap_adata'")

    X = [X_2d[adata.obs[samples_key] == t] for t in unique_times]
    bg_obs_values ="cell_type",
    if bg_obs_key is not None:
        if bg_obs_key not in adata.obs.columns:
            raise ValueError(f"bg_obs_key '{bg_obs_key}' not found in adata.obs")
        bg_obs_values = [
            adata.obs.loc[adata.obs[samples_key] == t, bg_obs_key].to_numpy()
            for t in unique_times
        ]

    np.save(os.path.join(output_path, "ode_traj_2d.npy"), traj_2d)
    np.save(os.path.join(output_path, "ode_point_2d.npy"), point_2d)
    print(f"[generate_ode_trajectories] trajectories saved to {output_path}")

    traj_times = np.linspace(float(sim_time_ticks[0]), float(sim_time_ticks[-1]), traj_2d.shape[0])
    output_path = os.path.join(output_path, save_base_name)
    plot_ode_v3_core(
        X=X,
        traj_array=traj_2d,
        save_path=output_path,
        traj_times=traj_times,
        time_ticks=sim_time_ticks,
        bg_obs_values=bg_obs_values,
        bg_obs_name=bg_obs_key,
        bg_palette=bg_palette,
        traj_cmap=traj_cmap,
        background_mode=background_mode,
        show_axes=show_axes,
    )

    print(f"[ode_trajectories_v2] done -> {output_path}")

# ==================== 修改后的 plot_sde_trajectories ====================
def plot_sde_trajectories(
        adata: sc.AnnData,
        output_path: str = "./sde_results",
        device: str = "cuda",
        sigma: float = 1.0,
        n_bins: int = 10,
        n_trajectories: int = 100,
        init_time: int = 0,
        dim_reduction: str = 'none',
        split_true: bool = False
):
    os.makedirs(output_path, exist_ok=True)
    device = torch.device(device)
    print(f"Using device: {device}")

    sde_traj, sde_point, w_point = generate_sde_trajectories(
        adata,
        exp_dir=output_path,
        device=device,
        sigma=sigma,
        n_time_steps=n_bins,
        sample_traj_num=n_trajectories,
        init_time=init_time,
        split_true=split_true
    )

    # Project to 2D Space
    dim_reduction = str(dim_reduction).lower()
    if dim_reduction == 'umap' or dim_reduction == 'umap_adata':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        plot_2d = adata.obsm['X_umap'][:, :2]
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            raise ValueError('Please run sc.pp.pca(adata) first')
        plot_2d = adata.obsm['X_pca'][:, :2]
    elif dim_reduction in {'none', 'null', ''}:
        plot_2d = adata.obsm['X_latent'][:, :2]
    else:
        raise ValueError(f"Invalid dim_reduction: {dim_reduction}")
      
    time_key = 'time_point_processed'
    real_df = pd.DataFrame(plot_2d, columns=['x1', 'x2'])
    real_df['samples'] = adata.obs[time_key].values
    D = sde_traj.shape[-1]
    # Project trajectories
    if D == 2:
        traj_2d = sde_traj
        point_2d = sde_point[..., :2]
    elif dim_reduction == 'pca':
        mean_ = adata.uns['pca']['mean'] if 'mean' in adata.uns['pca'] else 0.
        comps = adata.uns['pca']['components_'][:, :2]
        traj_2d = np.dot(sde_traj - mean_, comps)
        point_2d = np.dot(sde_point - mean_, comps)
    elif dim_reduction == 'umap':
        # 使用统一的UMAP接口
        umap_path = os.path.join(output_path, 'umap_model.pkl')
        reducer = get_or_train_umap(
            data=adata.obsm['X_latent'],
            save_path=umap_path,
            n_neighbors=10,
            min_dist=0.2,
            n_components=2,
            random_state=42
        )
        
        traj_2d = apply_umap_transform(reducer, sde_traj)
        point_2d = apply_umap_transform(reducer, sde_point)
    elif dim_reduction == 'umap_adata':
        # 获取traj_array和point_array的形状信息
        T_bins, M_1, _ = sde_traj.shape
        T, M_2, _ = sde_point.shape
  
        # 调整traj_array和sde_point的形状为二维
        traj_reshaped = sde_traj.reshape(-1, D)  # (T_bins*M, D)
        point_reshaped = sde_point.reshape(-1, D)  # (T*M, D)

        # 创建临时adata，包含所有需要降维的数据
        temp_adata = sc.AnnData(
            X=np.concatenate([adata.obsm['X_latent'], traj_reshaped, point_reshaped], axis=0)
        )
        # 存储原始数据的索引边界
        x_raw_end = len(adata.obsm['X_latent'])
        traj_end = x_raw_end + len(traj_reshaped)


        # 使用X_latent计算邻居并进行UMAP降维
        # temp_adata.obsm['X_latent'] = temp_adata.X.copy()  # 将数据存入X_latent
        # sc.pp.neighbors(temp_adata, use_rep='X_latent')
        # sc.tl.umap(temp_adata)
        
        sc.pp.neighbors(temp_adata)
        sc.tl.umap(temp_adata)
        
        # 提取降维后的结果
        #X_2d = temp_adata.obsm['X_umap'][:x_raw_end]  # (batchsize, 2)
        traj_2d_unshape = temp_adata.obsm['X_umap'][x_raw_end:traj_end]  # (T_bins*M_1, 2)
        point_2d_unshape = temp_adata.obsm['X_umap'][traj_end:]  # (T*M_2, 2)
        
        # 恢复原始形状
        traj_2d = traj_2d_unshape.reshape(T_bins, M_1, 2)  # (T_bins, M_1, 2)
        point_2d = point_2d_unshape.reshape(T, M_2, 2)  # (T, M_2, 2)
    else:
        traj_2d = sde_traj[..., :2]
        point_2d = sde_point[..., :2]

    print("Plotting SDE trajectories...")
    sde_plot(df=real_df,  
             generated=point_2d,
             trajectories=traj_2d,
             save=True,  
             output_path=output_path,  
             file='sde_trajectories.pdf')

    return adata
    
from CytoBridge.Map.tl.Mongemap import TransportMap

def plot_ode_trajectories_map(
    adata,
    model,
    tranmap,
    output_path='figures',
    save_base_name='ode_trajectories_map.pdf',
    n_trajectories=50,
    n_bins=100,
    dim_reduction='umap',
    device="cuda",
    split_true=False
):
    model.to(device)
    model.eval()
    output_path = os.path.join(output_path, "ode_results")
    os.makedirs(output_path, exist_ok=True)

    point_array, traj_array = generate_ode_trajectories(
        model, adata,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
        split_true=split_true
    )

    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())
    X_raw = adata.obsm['X_latent']
    D = traj_array.shape[-1]

    point_array = point_array.reshape(-1, D)
    traj_array = traj_array.reshape(-1, D)

    # 将 X_raw 转换为 torch.Tensor 并应用 tranmap.T
    X_raw = torch.tensor(X_raw, dtype=torch.float32, device=device)
    X_raw = tranmap.T(X_raw).cpu().numpy()  # 应用变换并移回 CPU

    # 将 point_array 和 traj_array 转换为 torch.Tensor
    point_array = torch.tensor(point_array, dtype=torch.float32, device=device)
    traj_array = torch.tensor(traj_array, dtype=torch.float32, device=device)

    # 应用 tranmap.T 并确保结果在 CPU 上
    point_array = tranmap.T(point_array).cpu().numpy()
    traj_array = tranmap.T(traj_array).cpu().numpy()

    # point_array = (point_array ** 3).cpu().numpy()
    # traj_array = (traj_array ** 3).cpu().numpy()
    D =  point_array.shape[-1]
    # 重新变回原来的形状
    point_array = point_array.reshape(-1, n_trajectories,D)
    traj_array = traj_array.reshape(-1, n_trajectories, D)


    # 2. Reduce dimension to 2-D
    if D == 2:
        traj_2d, point_2d = traj_array, point_array
        X_2d = X_raw
    else:
        reducer_pkl = os.path.join(output_path, "dim_reducer_map.pkl")

        if dim_reduction == 'none':
            traj_2d  = traj_array[..., :2]
            point_2d = point_array[..., :2]
            X_2d     = X_raw[..., :2]
        elif dim_reduction in ('pca', 'umap'):
            # --- First check if there is an existing reducer locally ---
            if os.path.isfile(reducer_pkl):
                with open(reducer_pkl, 'rb') as f:
                    reducer = pickle.load(f)
                print(f"[dim-reduction] loaded existing reducer from {reducer_pkl}")
                # For safety, check dimension
                if hasattr(reducer, 'n_components') and reducer.n_components != 2:
                    raise RuntimeError("Cached reducer n_components != 2, please delete it manually.")
            else:
                print(f"[dim-reduction] no cached reducer, training {dim_reduction.upper()} ...")
                if dim_reduction == 'pca':
                    
                    reducer = PCA(n_components=2, random_state=0)
                else:  # umap
                    
                    reducer = umap.UMAP(n_components=2, random_state=0)
                reducer.fit(X_raw)
                # Save for future reuse
                with open(reducer_pkl, 'wb') as f:
                    pickle.dump(reducer, f)
                print(f"[dim-reduction] reducer saved to {reducer_pkl}")

            # --- Uniform dimensionality reduction ---
            X_2d     = reducer.transform(X_raw)
            traj_2d  = reducer.transform(traj_array.reshape(-1, D)).reshape(*traj_array.shape[:2], 2)
            point_2d = reducer.transform(point_array.reshape(-1, D)).reshape(*point_array.shape[:2], 2)

        elif dim_reduction == 'umap_adata':
            # 获取traj_array和point_array的形状信息
            T_bins, M, _ = traj_array.shape
            T, _, _ = point_array.shape
            
            # 调整traj_array和point_array的形状为二维
            traj_reshaped = traj_array.reshape(-1, D)  # (T_bins*M, D)
            point_reshaped = point_array.reshape(-1, D)  # (T*M, D)
            
            # 创建临时adata，包含所有需要降维的数据
            temp_adata = sc.AnnData(
                X=np.concatenate([X_raw, traj_reshaped, point_reshaped], axis=0)
            )
            # 存储原始数据的索引边界
            x_raw_end = len(X_raw)
            traj_end = x_raw_end + len(traj_reshaped)
            
            # 使用X_latent计算邻居并进行UMAP降维
            sc.pp.neighbors(temp_adata)
            sc.tl.umap(temp_adata)
            # temp_adata.obsm['X_latent'] = temp_adata.X.copy()  # 将数据存入X_latent
            # sc.pp.neighbors(temp_adata, use_rep='X_latent')
            # sc.tl.umap(temp_adata)
            # 提取降维后的结果
            X_2d = temp_adata.obsm['X_umap'][:x_raw_end]  # (batchsize, 2)
            traj_2d_unshape = temp_adata.obsm['X_umap'][x_raw_end:traj_end]  # (T_bins*M, 2)
            point_2d_unshape = temp_adata.obsm['X_umap'][traj_end:]  # (T*M, 2)
            
            # 恢复原始形状
            traj_2d = traj_2d_unshape.reshape(T_bins, M, 2)  # (T_bins, M, 2)
            point_2d = point_2d_unshape.reshape(T, M, 2)  # (T, M, 2)
        else:
            raise ValueError("dim_reduction must be 'none', 'pca', 'umap' or 'umap_adata'")

        # Slice by time
    X = [X_2d[adata.obs[samples_key] == t] for t in unique_times]

    # 3. Save & plot
    # np.save(os.path.join(output_path, "ode_traj.npy"),  traj_2d)
    # np.save(os.path.join(output_path, "ode_point.npy"), point_2d)
    print(f"[generate_ode_trajectories_map] trajectories saved to {output_path}")

    output_path = os.path.join(output_path, save_base_name)
    plot_ode(X=X,
             point_array=point_2d,
             traj_array=traj_2d,
             save_dir=output_path)

    print(f"[ode_trajectories_map] done -> {output_path}")


def _apply_transport_map_array(tranmap, array, device="cuda"):
    array_np = np.asarray(array)
    lead_shape = array_np.shape[:-1]
    flat = array_np.reshape(-1, array_np.shape[-1])
    flat_t = torch.as_tensor(flat, dtype=torch.float32, device=device)

    if hasattr(tranmap, "T"):
        mapped = tranmap.T(flat_t)
    elif hasattr(tranmap, "map"):
        mapped = tranmap.map(flat_t)
    else:
        raise ValueError("tranmap must provide either .T(tensor) or .map(tensor)")

    if torch.is_tensor(mapped):
        mapped_np = mapped.detach().cpu().numpy()
    else:
        mapped_np = np.asarray(mapped)
    if mapped_np.ndim != 2:
        raise ValueError(f"Mapped result must be 2D, got {mapped_np.shape}")
    return mapped_np.reshape(*lead_shape, mapped_np.shape[-1])


def plot_ode_trajectories_map_v2(
    adata,
    model,
    tranmap,
    output_path='figures',
    save_base_name='ode_trajectories_map_v2.pdf',
    n_trajectories=50,
    n_bins=100,
    dim_reduction='umap',
    bg_obs_key="time_point_processed",
    bg_palette="Set2",
    traj_cmap="sunset1",
    background_mode="auto",
    show_axes=True,
    device="cuda",
    split_true=False,
    sampling_mode="random",
    init_time=None,
    init_cell_type=None,
    init_indices=None,
    cell_type_key="cell_type",
    sampling_seed=None,
    strict_index_init_time=True,
    perturbation_cfg=None,
    perturbation_reference_adata=None,
):
    model.to(device)
    model.eval()
    output_path = os.path.join(output_path, "ode_results")
    os.makedirs(output_path, exist_ok=True)

    point_array, traj_array, sampling_meta = generate_ode_trajectories(
        model, adata,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
        exp_dir=output_path,
        split_true=split_true,
        sampling_mode=sampling_mode,
        init_time=init_time,
        init_cell_type=init_cell_type,
        init_indices=init_indices,
        cell_type_key=cell_type_key,
        sampling_seed=sampling_seed,
        strict_index_init_time=strict_index_init_time,
        perturbation_cfg=perturbation_cfg,
        perturbation_reference_adata=perturbation_reference_adata,
        return_metadata=True,
    )

    samples_key = 'time_point_processed'
    unique_times = np.sort(np.asarray(adata.obs[samples_key].unique(), dtype=float))
    sim_time_ticks = np.asarray(sampling_meta.get("time_points", unique_times), dtype=float)
    if sim_time_ticks.size == 0:
        sim_time_ticks = unique_times
    X_raw = adata.obsm['X_latent']
    print()
    X_map = _apply_transport_map_array(tranmap, X_raw, device=device)
    point_map = _apply_transport_map_array(tranmap, point_array, device=device)
    traj_map = _apply_transport_map_array(tranmap, traj_array, device=device)
    D_map = traj_map.shape[-1]
    reduction_mode = str(dim_reduction).lower()

    if D_map == 2:
        traj_plot, point_plot, X_plot = traj_map, point_map, X_map
    elif D_map == 3 and reduction_mode in {'none', 'null', ''}:
        traj_plot, point_plot, X_plot = traj_map, point_map, X_map
    else:
        if reduction_mode in {'none', 'null', ''}:
            traj_plot = traj_map[..., :2]
            point_plot = point_map[..., :2]
            X_plot = X_map[..., :2]
        elif reduction_mode in ('pca', 'umap'):
            reducer_pkl = os.path.join(output_path, f"dim_reducer_map_v2_{reduction_mode}.pkl")

            if reduction_mode == 'pca':
                if os.path.isfile(reducer_pkl):
                    with open(reducer_pkl, 'rb') as f:
                        reducer = pickle.load(f)
                    print(f"[dim-reduction] loaded existing PCA from {reducer_pkl}")
                else:
                    print("[dim-reduction] training PCA...")
                    reducer = PCA(n_components=2, random_state=0)
                    reducer.fit(X_map)
                    with open(reducer_pkl, 'wb') as f:
                        pickle.dump(reducer, f)
                    print(f"[dim-reduction] PCA saved to {reducer_pkl}")
            else:
                reducer = get_or_train_umap(
                    data=X_map,
                    save_path=reducer_pkl,
                    n_neighbors=50,
                    min_dist=0.2,
                    n_components=2,
                    random_state=42
                )

            X_plot = fitted_umap_background(reducer, X_map)
            traj_plot = apply_umap_transform(reducer, traj_map)
            point_plot = apply_umap_transform(reducer, point_map)
        elif reduction_mode == 'umap_adata':
            T_bins, M, _ = traj_map.shape
            T, _, _ = point_map.shape

            traj_reshaped = traj_map.reshape(-1, D_map)
            point_reshaped = point_map.reshape(-1, D_map)

            temp_adata = sc.AnnData(
                X=np.concatenate([X_map, traj_reshaped, point_reshaped], axis=0)
            )
            x_raw_end = len(X_map)
            traj_end = x_raw_end + len(traj_reshaped)

            sc.pp.neighbors(temp_adata, n_neighbors=40, random_state=42)
            sc.tl.umap(temp_adata, random_state=42)
            X_plot = temp_adata.obsm['X_umap'][:x_raw_end]
            traj_unshape = temp_adata.obsm['X_umap'][x_raw_end:traj_end]
            point_unshape = temp_adata.obsm['X_umap'][traj_end:]

            traj_plot = traj_unshape.reshape(T_bins, M, 2)
            point_plot = point_unshape.reshape(T, M, 2)
        else:
            raise ValueError("dim_reduction must be 'none', 'pca', 'umap' or 'umap_adata'")

    X = [X_plot[adata.obs[samples_key] == t] for t in unique_times]
    bg_obs_values = None
    if bg_obs_key is not None:
        if bg_obs_key not in adata.obs.columns:
            raise ValueError(f"bg_obs_key '{bg_obs_key}' not found in adata.obs")
        bg_obs_values = [
            adata.obs.loc[adata.obs[samples_key] == t, bg_obs_key].to_numpy()
            for t in unique_times
        ]

    np.save(os.path.join(output_path, "ode_traj_map.npy"), traj_map)
    np.save(os.path.join(output_path, "ode_point_map.npy"), point_map)
    np.save(os.path.join(output_path, "ode_x_map_plot.npy"), X_plot)
    np.save(os.path.join(output_path, "ode_traj_map_plot.npy"), traj_plot)
    np.save(os.path.join(output_path, "ode_point_map_plot.npy"), point_plot)
    print(f"[generate_ode_trajectories_map_v2] trajectories saved to {output_path}")

    traj_times = np.linspace(float(sim_time_ticks[0]), float(sim_time_ticks[-1]), traj_plot.shape[0])
    save_path = os.path.join(output_path, save_base_name)
    plot_ode_v3_core(
        X=X,
        traj_array=traj_plot,
        save_path=save_path,
        traj_times=traj_times,
        time_ticks=sim_time_ticks,
        bg_obs_values=bg_obs_values,
        bg_obs_name=bg_obs_key,
        bg_palette=bg_palette,
        traj_cmap=traj_cmap,
        background_mode=background_mode,
        show_axes=show_axes,
    )

    print(f"[ode_trajectories_map_v2] done -> {save_path}")
#%%
import CytoBridge
import math

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'axes.spines.right': False,
    'axes.spines.top': False,
    'font.size': 12,
    'axes.labelsize': 12
})

def plot_score_and_gradient(dynamical_model, device, t_value=1.0, x_range=(0, 2), y_range=(0, 2.5),
                            step=5, output_path=None, cmap='rainbow'):
    """
    Plot the output log probability of the score component in DynamicalModel and its gradient vector field (adapted for scoreNet2)
    """
    device = torch.device(device)          # Ensure it's a torch.device object
    dynamical_model = dynamical_model.to(device)   # ← Critical: Move model to device
    if 'score' not in dynamical_model.components:
        raise ValueError("DynamicalModel must contain 'score' component")

    if dynamical_model.velocity_net.input_layer[0].in_features !=3:
        raise ValueError("DynamicalModel's input (gene expression) must be 2-dimensional")
        
    # Create grid (100x100 grid)
    grid_size = 100
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # Shape: (10000, 2)

    # Convert to tensor (separate time and position, adapted to input format of scoreNet2)
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    t_value = torch.tensor([t_value], dtype=torch.float32, device=device).unsqueeze(1)
    log_density, gradients = dynamical_model.compute_score(t_value, x_tensor)
    
    # Convert to numpy and reshape to grid shape
    log_density_np = log_density.cpu().detach().numpy().reshape(grid_size, grid_size)  # 100x100
    density = torch.exp(log_density).cpu().detach().numpy().reshape(grid_size, grid_size)
    gradients_np = gradients.cpu().detach().numpy().reshape(grid_size, grid_size, 2)  # 100x100x2

    # Sample gradient vectors (downsample to avoid overcrowding)
    xv_quiver = xv[::step, ::step]
    yv_quiver = yv[::step, ::step]
    grad_quiver = gradients_np[::step, ::step, :]  # Sampled gradients


    # Plotting
    plt.figure(figsize=(8, 6))
    # Plot negative log probability contours
    contour = plt.contourf(
        xv, yv, -log_density_np,  # Use negative log probability
        levels=50,
        cmap=cmap,
        alpha=0.8
    )
    plt.colorbar(contour, label='-log density')

    # Plot gradient vector field
    plt.quiver(
        xv_quiver, yv_quiver,
        grad_quiver[:, :, 0], grad_quiver[:, :, 1],
        color='white',
        scale=1,
    )

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f't = {t_value}')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
    else:
        plt.show()
    plt.close()


#%%


import torch

from torch.nn import Module


from CytoBridge.tl.analysis import generate_sde_trajectories


# ------------------------------------------------------------------------------
# Plotting Function Dependencies: Matplotlib, Linear Segmented Colormap


def sde_plot(df, generated, trajectories, palette='viridis',
             df_time_key='samples', x='x1', y='x2',
             save=False, output_path="./sde_results", file='sde_trajectories.pdf'):
    """
    Plot SDE trajectories with ground truth data, predicted points, and trajectory lines.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of ground truth data with columns for x/y coordinates and time.
    generated : np.ndarray
        SDE-predicted discrete points (shape: [n_time_points, batch_size, 2]).
    trajectories : np.ndarray
        SDE-simulated continuous trajectories (shape: [n_time_steps, batch_size, 2]).
    palette : str, default='viridis'
        Matplotlib colormap name for time-based coloring.
    df_time_key : str, default='samples'
        Column name in df containing time labels.
    x : str, default='x1'
        Column name in df for x-axis coordinates.
    y : str, default='x2'
        Column name in df for y-axis coordinates.
    save : bool, default=False
        Whether to save the plot to file.
    output_path : str, default="./sde_results"
        Directory to save the plot (only used if save=True).
    file : str, default='sde_trajectories.pdf'
        Filename for the saved plot (only used if save=True).
    
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib Figure object of the SDE trajectory plot.
    """
    # Get sorted unique time groups
    groups = sorted(df[df_time_key].unique())
    # Initialize custom colormap
    cmap = plt.colormaps.get_cmap(palette)
    new_cmap = LinearSegmentedColormap.from_list('viridis_custom',
                                                 cmap(np.linspace(0, 1, 256)))

    # Configure plot aesthetics
    plt.rcParams.update({
        'axes.edgecolor': 'lightgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'xtick.bottom': False,
        'ytick.left': False,
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 10
    })

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)

    # 1. Plot ground truth data points (X markers)
    ax.scatter(df[x], df[y], c=df[df_time_key], s=100, alpha=0.5,
               marker='X', cmap=new_cmap, label='Ground Truth')

    # 2. Plot SDE trajectories (thin red lines with low opacity to avoid overcrowding)
    # Transpose trajectories to (batch_size, n_time_steps, 2) for per-trajectory iteration
    for traj in np.transpose(trajectories, (1, 0, 2)):
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.4, color='firebrick', lw=1)

    # 3. Plot SDE-predicted discrete points (O markers with double-layer coloring for contrast)
    pred_points = np.concatenate(generated, axis=0)  # Merge predicted points across all time steps
    # Match time labels to predicted points (repeat time for all points in the same time step)
    pred_times = np.concatenate([[t] * generated[i].shape[0]
                                 for i, t in enumerate(groups)])
    # Outer black layer (enhances visibility against background)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c='black',
               s=120, alpha=0.8, marker='o')
    # Inner colored layer (encodes time information)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c=pred_times,
               s=100, alpha=0.8, marker='o', cmap=new_cmap, label='SDE Predicted')

    # 4. Build legends (time labels + data type)
    # Legend for time points (top-left)
    time_legend = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=new_cmap(i / len(groups)), markersize=12)
                   for i, t in enumerate(groups)]
    leg1 = ax.legend(handles=time_legend, loc='upper left', title='Time Points')
    ax.add_artist(leg1)  # Preserve time legend (avoid being overwritten by subsequent legends)

    # Legend for data types (top-right)
    type_legend = [
        Line2D([0], [0], marker='X', color='w', label='Ground Truth',
               markerfacecolor='grey', markersize=12, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='SDE Predicted',
               markerfacecolor='grey', markersize=12),
        Line2D([0], [0], color='red', lw=2, label='SDE Trajectory', alpha=0.5)
    ]
    ax.legend(handles=type_legend, loc='upper right')

    # 5. Set axis labels and plot title
    ax.set_xlabel("Latent Dimension 1", fontsize=14)
    ax.set_ylabel("Latent Dimension 2", fontsize=14)
    ax.set_title("SDE Trajectories (Combining Velocity/Growth/Score)", fontsize=16, pad=20)

    # 6. Save plot if enabled (adjust layout to prevent label cutoff)
    if save:
        plt.tight_layout()
        # Create directory if it doesn't exist (defensive check)
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.join(output_path, file), bbox_inches='tight')
        print(f"SDE trajectory plot saved to: {os.path.join(output_path, file)}")
    fig.show()

    # Close plot to free memory (avoids unintended memory leaks in batch runs)
    #plt.close()

    return fig

#%%
import anndata
import scvelo as scv



from CytoBridge.tl.analysis import compute_velocity,compute_velocity_map

def plot_velocity_stream(adata, model,output_path, dim_reduction='none', device='cuda'):
    """
    Plot latent space velocity field streamlines
    Coordinates always use adata.obsm['X_latent'][:, :2]
    Velocity uses adata.layers['velocity_latent']
    If velocity is not 2-D, perform projection using its own basis
    """
    # 1. Ensure velocity has been calculated
    if 'velocity_latent' not in adata.layers:
        adata = compute_velocity(adata, model,device)

    # 2. Prepare coordinates (2-D)
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2)
        adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].copy()
    else:  # 'none' directly uses first two dimensions of latent space
        adata.obsm['X_umap'] = adata.obsm['X_latent'][:, :2].copy()

    # 3. Place velocity in scVelo's specified key
    adata.layers['Ms']   = adata.obsm['X_latent'].copy()
    adata.layers['velocity'] = adata.obsm['velocity_latent']

    # 4. 2-D velocity processing
    if adata.layers['velocity'].shape[1] == 2:
        # Velocity is already 2-D, directly use as umap velocity
        adata.obsm['velocity_umap'] = adata.layers['velocity']
    else:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')  # Calculate neighbors based on high-dim data
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)  # Build velocity graph from high-dim velocity vectors
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')  # Project 

    # 5. Time coloring
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])

    # 6. Plotting
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))

    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title='Velocity Stream Plot',
        legend_loc='right',
        palette='plasma',
        save='Velocity_Stream_Plot.svg',
    )
    print(f"Velocity stream plot saved to: {output_path}/Velocity_Stream_Plot.svg")
    
    if 'cell_type' in adata.obs.columns :
        scv.pl.velocity_embedding_stream(
            adata,
            basis='umap',
            color='cell_type',
            figsize=(7, 5),
            density=3,
            title='Velocity Stream Plot',
            legend_loc='right',
            palette='plasma',
            save='Velocity_cluster_Stream_Plot.svg',
        )

    return adata

import scvelo as scv

from CytoBridge.tl.analysis import compute_map_X_latent, compute_velocity_map

def plot_velocity_stream_map(adata, model, tranmap, output_path, dim_reduction='none', device='cuda'):
    """
    基于转换后的潜在空间和次空间速度绘制速度流图
    
    参数:
        adata: AnnData对象，包含潜在空间和时间信息
        model: 训练好的模型
        tranmap: TransportMap实例
        output_path: 图像保存目录
        dim_reduction: 降维方法（'umap'/'pca'/'none'）
        device: 计算设备
    """
    # 检查并计算次空间速度
    if 'velocity_map_latent' not in adata.obsm:
        adata = compute_velocity_map(adata, model, tranmap, device=device)
    
    # 检查并计算转换后的潜在空间
    if 'X_latent_map' not in adata.obsm:
        adata = compute_map_X_latent(adata, tranmap,device)
    
    # 创建临时AnnData用于降维计算
    temp_adata = sc.AnnData(
        X=adata.obsm['X_latent_map'],
        obs=adata.obs.copy(),
        layers={
            'velocity': adata.obsm['velocity_map_latent'].copy()
        }
    )
    
    # 准备2D坐标
    if dim_reduction == 'umap':
        if 'X_umap' not in temp_adata.obsm:
            sc.pp.neighbors(temp_adata)
            sc.tl.umap(temp_adata)
        temp_adata.obsm['X_umap'] = temp_adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in temp_adata.obsm:
            sc.pp.pca(temp_adata, n_comps=2)
        temp_adata.obsm['X_umap'] = temp_adata.obsm['X_pca'][:, :2].copy()
    else:  # 'none'使用潜在空间前两维
        temp_adata.obsm['X_umap'] = temp_adata.X[:, :2].copy()
    
    # 设置scVelo所需的层
    temp_adata.layers['Ms'] = temp_adata.X.copy()  # 模拟scVelo的spliced层
    
    # 处理2D速度
    if temp_adata.layers['velocity'].shape[1] == 2:
        temp_adata.obsm['velocity_umap'] = temp_adata.layers['velocity']
    else:
        sc.pp.neighbors(temp_adata, n_neighbors=30, use_rep='X')
        scv.tl.velocity_graph(temp_adata, vkey='velocity', n_jobs=16)
        scv.tl.velocity_embedding(temp_adata, basis='umap', vkey='velocity')
    
    # 时间分组
    temp_adata.obs['time_categorical'] = pd.Categorical(temp_adata.obs['time_point_processed'])
    
    # 绘图设置
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))
    
    # 绘制速度流图
    scv.pl.velocity_embedding_stream(
        temp_adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title='Velocity Stream Plot (Mapped Space)',
        legend_loc='right',
        palette='plasma',
        save='Velocity_Stream_Map_Plot.svg',
    )
    scv.pl.velocity_embedding_stream(
        temp_adata,
        basis='umap',
        color='cell_type',
        figsize=(7, 5),
        density=3,
        title='Velocity Stream Plot (Mapped Space)',
        legend_loc='right',
        palette='plasma',
        save='Velocity_cluster_Stream_Map_Plot.svg',
    )
    print(f"[plot_velocity_stream_map] has been saved -> {output_path}")
#%%
import anndata
import scvelo as scv



import torch
from CytoBridge.tl.analysis import compute_interaction_force  # Import interaction calculation function

def plot_interaction_stream(adata, output_path, dim_reduction='none', device='cuda'):
    """
    Plot interaction force field streamlines
    Coordinates use first two dimensions of adata.obsm['X_latent'] or dimensionality reduction results
    Force vectors are calculated from the model's interaction component
    """
    # 1. Ensure interaction force has been calculated
    if 'interaction_force' not in adata.layers:
        adata = compute_interaction_force(adata, device)
    
    # 2. Check required data
    required_keys = ['X_latent', 'time_point_processed']
    for key in required_keys:
        if (key not in adata.obsm) and (key not in adata.obs):
            raise ValueError(f"Required data not found: {key}")
    
    # 3. Prepare 2D coordinates
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_latent')
            sc.tl.umap(adata)
        coords_2d = adata.obsm['X_umap']
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2, use_rep='X_latent')
        coords_2d = adata.obsm['X_pca'][:, :2]
    else:  # Use first two dimensions of latent space
        coords_2d = adata.obsm['X_latent'][:, :2].copy()
    
    # 4. Prepare interaction force vectors
    force = adata.layers['interaction_force']
    # print(force.shape)
    # print(force)
    # 5. Process force vectors for 2D visualization
    if force.shape[1] == 2:
        # Force is already 2D
        force_2d = force
    else:
        # Project high-dimensional force to 2D using the same reduction
        if dim_reduction == 'umap':
            reducer = UMAP().fit(adata.obsm['X_latent'])
            data_end = adata.obsm['X_latent'] + force
            data_end_2d = reducer.transform(data_end)
            force_2d = data_end_2d - coords_2d
        elif dim_reduction == 'pca':
            
            pca = PCA(n_components=2).fit(adata.obsm['X_latent'])
            data_end = adata.obsm['X_latent'] + force
            data_end_2d = pca.transform(data_end)
            force_2d = data_end_2d - coords_2d
        else:
            # Directly use first two dimensions for 'none'
            force_2d = force[:, :2]

    # 6. Normalize force vectors for visualization
    force_2d = force_2d / np.linalg.norm(force_2d, axis=1, keepdims=True) * 5
    
    # 7. Prepare scVelo required format
    adata.obsm['X_umap'] = coords_2d
    adata.layers['Ms'] = adata.obsm['X_latent'].copy()
    adata.layers['velocity'] = force  # Use interaction force as "velocity" for scVelo
    # Handle high-dimensional force projection if needed
    if force_2d.shape[1] == 2:
        adata.obsm['velocity_umap'] = force_2d
    else:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_latent')
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')
    
    # 8. Time information for coloring
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])
    
    # 9. Plot settings
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))


    # 10. Generate and save plot
    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title='Interaction Force Stream Plot',
        legend_loc='right',
        palette='plasma',
        save='Interaction_Force_Stream_Plot.svg',
    )
    print(f"Interaction force stream plot saved to: {output_path}/Interaction_Force_Stream_Plot.svg")
    return adata
#%%
import anndata
import scvelo as scv



import torch

def plot_score_stream(adata,model,output_path, dim_reduction='none', device='cuda'):
    """
    Plot score function diffusion velocity field streamlines
    Coordinates use first two dimensions of adata.obsm['X_latent'] or dimensionality reduction results
    Velocity is calculated from the gradient of the score function (log density)
    """
    # 1. Check if necessary data exists
    required_keys = ['X_latent', 'time_point_processed']
    for key in required_keys:
        if (key not in adata.obsm) and (key not in adata.obs):
            raise ValueError(f"Required data not found: {key}, please ensure the model has correctly computed relevant results")

    # 2. Prepare input data (latent space data and time)
    device = torch.device(device)
    all_data = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    all_data.requires_grad_(True)  # Enable gradient tracking (critical modification)
    all_times = torch.tensor(adata.obs['time_point_processed'].values, dtype=torch.float32, device=device).unsqueeze(1)

    # 3. Calculate score function (log density) and gradients (velocity vectors)
    # Calculate log density values directly from the model, ensuring complete computation graph
    model.to(device)
    model.eval()
    log_density_values, gradients = model.compute_score(all_times, all_data)  # Directly compute with model (critical modification)

    # 4. Process coordinate and velocity dimensionality reduction
    data_np = all_data.detach().cpu().numpy()
    gradients_np = gradients.detach().cpu().numpy()

    # Prepare 2D coordinates
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_latent')
            sc.tl.umap(adata)
        data_np_2d = adata.obsm['X_umap']
        # Perform same dimensionality reduction on gradients
        data_end = data_np + gradients_np
        reducer = UMAP().fit(adata.obsm['X_latent'])
        data_end_2d = reducer.transform(data_end)
        gradients_np_2d = data_end_2d - data_np_2d
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2, use_rep='X_latent')
        data_np_2d = adata.obsm['X_pca'][:, :2]
        # Perform same dimensionality reduction on gradients
        
        pca = PCA(n_components=2).fit(adata.obsm['X_latent'])
        data_end = data_np + gradients_np
        data_end_2d = pca.transform(data_end)
        gradients_np_2d = data_end_2d - data_np_2d
    else:  # Directly use first two dimensions of latent space
        data_np_2d = data_np[:, :2]
        gradients_np_2d = gradients_np[:, :2]

    # 5. Normalize velocity vectors (uniform length for visualization)
    gradients_np_2d = gradients_np_2d / np.linalg.norm(gradients_np_2d, axis=1, keepdims=True) * 5

    # 6. Prepare data format required by scVelo
    adata.obsm['X_umap'] = data_np_2d  # 2D coordinates (compatible with scVelo's basis='umap')
    adata.layers['Ms'] = data_np  # Original high-dimensional state matrix
    adata.layers['velocity'] = gradients_np  # High-dimensional velocity vectors
    if gradients_np_2d.shape[1] == 2:
        adata.obsm['velocity_umap'] = gradients_np_2d  # Directly use 2D velocity
    else:
        # High-dimensional velocity needs neighbor graph construction and projection to 2D coordinates
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_latent')
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')

    # 7. Time information processing
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])

    # 8. Plot settings and saving
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))

    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title='Score Stream Plot',
        legend_loc='right',
        palette='plasma',
        save='Score_Stream_Plot.svg',
    )
    print(f"Score function diffusion velocity stream plot saved to: {output_path}/Score_Stream_Plot.svg")
    return adata
#%%
import anndata
import scvelo as scv



import torch

def plot_combined_velocity_stream(adata,model,output_path, dim_reduction='none', device='cuda'):
    """
    Plot combined velocity stream plot (model velocity + score gradient velocity)
    
    Parameters:
        - alpha: weight for model velocity
        - beta: weight for score gradient velocity
    """
    alpha=0.5
    beta=0.5
    # --------------------------
    # 1. Calculate and obtain both velocities
    # --------------------------
    # 1.1 Obtain model velocity (velocity_latent)
    if 'velocity_latent' not in adata.layers:
        from CytoBridge.tl.analysis import compute_velocity
        adata = compute_velocity(adata, model,device)
    vel = adata.obsm['velocity_latent'].copy()  # Model velocity
    norms = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_model = np.divide(vel, norms, out=np.zeros_like(vel), where=norms!=0) * 5
    # 1.2 Calculate score gradient velocity
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    # Prepare input data
    all_data = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    all_data.requires_grad_(True)
    all_times = torch.tensor(
        adata.obs['time_point_processed'].values, 
        dtype=torch.float32, 
        device=device
    ).unsqueeze(1)
    
    # Calculate score gradient (velocity)
    _, grad_score = model.compute_score(all_times, all_data)
    vel_score1 = grad_score.detach().cpu().numpy()  # Score gradient velocity
    norms = np.linalg.norm(vel_score1, axis=1, keepdims=True)
    vel_score = np.divide(vel_score1, norms, out=np.zeros_like(vel), where=norms!=0)
    # --------------------------
    # 2. Combine both velocities (weighted sum, maintaining original calculation method)
    # --------------------------
    combined_vel = alpha * vel_model + beta * vel_score  # Consistent with overall calculation logic in example
    adata.layers['combined_velocity'] = combined_vel  # Store combined velocity

    # --------------------------
    # 3. Prepare coordinates (2D), maintaining same dimensionality reduction logic as existing plotting functions
    # --------------------------
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_latent')
            sc.tl.umap(adata)
        adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2, use_rep='X_latent')
        adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].copy()
    else:  # Directly use first two dimensions of latent space
        adata.obsm['X_umap'] = adata.obsm['X_latent'][:, :2].copy()

    # --------------------------
    # 4. Adapt to scVelo format, maintaining same projection logic as existing functions
    # --------------------------
    adata.layers['Ms'] = adata.obsm['X_latent'].copy()  # Original high-dimensional state
    adata.layers['velocity'] = adata.layers['combined_velocity']  # Combined velocity

    # Process velocity projection (judge based on dimensions, consistent with logic in plot_velocity_stream/plot_score_stream)
    if combined_vel.shape[1] == 2:
        adata.obsm['velocity_umap'] = combined_vel  # Directly use 2D velocity
    else:
        # High-dimensional velocity needs neighbor graph construction and projection to 2D
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_latent')
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')

    # --------------------------
    # 5. Plot settings, maintaining same style as existing functions
    # --------------------------
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))

    # Plot combined velocity stream
    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title=f'Combined Velocity Stream',
        legend_loc='right',
        palette='plasma',
        save='All_Velocity_Stream.svg',
    )
    print(f"All velocity stream plot saved to: {output_path}/All_Velocity_Stream.svg")
    return adata
    #%%

import torch


def plot_interaction_potential(
    model,
    d=200,
    num_points=100,
    output_path=None,
    device="cuda"
):
    """
    Visualize interaction potential energy (input is distance d, output is energy value)
    
    Parameters:
        model: Trained model containing interaction network
        d: Range parameter for distance (d_min = -d, d_max = d)
        num_points: Number of sampling points
        output_path: Path to save the plot (do not save if None)
        device: Computing device
    """
    d_min = -d
    d_max = d
    # Load model
    model.to(device)
    model.eval()
    # Initialize ODE function (used for energy calculation)
    
    # Generate distance sequence
    d_values = np.linspace(d_min, d_max, num_points)
    d_tensor = torch.tensor(d_values, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension to match network input
    # Calculate energy (use no_grad to speed up and avoid gradient computation)
    with torch.no_grad():
        energies = model.interaction_net(d_tensor)
        # Critical fix: Move CUDA tensor to CPU and convert to NumPy array
        energies = energies.cpu().numpy().flatten()  # Flatten to 1D array to match d_values shape
    print(energies)

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, energies, 'b-', linewidth=2)
    plt.xlabel('Distance (d)')
    plt.ylabel('Interaction Potential Energy')
    plt.title('Interaction Potential vs Distance')
    plt.grid(True, alpha=0.3)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Interaction potential plot saved to: {output_path}")
    else:
        plt.show()

def plot_interaction_potential_epoch(
    model,
    d=200,
    num_points=100,
    output_path=None,
    device="cuda"
):
    """
    Visualize interaction potential energy (input is distance d, output is energy value)
    
    Parameters:
        model: Trained model containing interaction network
        d: Range parameter for distance (d_min = -d, d_max = d)
        num_points: Number of sampling points
        output_path: Path to save the plot (do not save if None)
        device: Computing device
    """
    d_min = -d
    d_max = d
    # Load model
    model.to(device)
    model.eval()
    # Initialize ODE function (used for energy calculation)
    
    # Generate distance sequence
    d_values = np.linspace(d_min, d_max, num_points)
    d_tensor = torch.tensor(d_values, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension to match network input
    # Calculate energy (use no_grad to speed up and avoid gradient computation)

    with torch.no_grad():
        energies = model.interaction_net(d_tensor)
        # Critical fix: Move CUDA tensor to CPU and convert to NumPy array
        energies = energies.cpu().numpy().flatten()  # Flatten to 1D array to match d_values shape
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, energies, 'b-', linewidth=2)
    plt.xlabel('Distance (d)')
    plt.ylabel('Interaction Potential Energy')
    plt.title('Interaction Potential vs Distance')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        last_slash_index = output_path.rfind('/')
        new_path = output_path[:last_slash_index]
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print(f"Directory created: {new_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Interaction potential plot saved to: {output_path}")
    else:
        plt.show()
import torch
import torch.nn as nn
import torch.optim as optim



import scvelo as scv
from CytoBridge.tl.analysis import compute_map_X_latent, compute_velocity_map

def plot_landscape_map(adata, model, tranmap, output_path=None, dim_reduction="none", device="cuda"):
    """
    基于转换后的潜在空间（X_latent_map）和次空间速度（velocity_map_latent）绘制势能景观图
    
    参数:
        adata: AnnData对象，包含潜在空间和速度信息
        model: 训练好的模型
        tranmap: TransportMap实例，用于空间转换
        output_path: 图像保存路径（None则不保存）
        dim_reduction: 降维方法（'umap'/'pca'/'none'）
        device: 计算设备
    """
    # 1. 检查并计算次空间速度
    if 'velocity_map_latent' not in adata.obsm:
        adata = compute_velocity_map(adata, model, tranmap, device=device)
    
    # 2. 检查并计算转换后的潜在空间
    if 'X_latent_map' not in adata.obsm:
        adata = compute_map_X_latent(adata, tranmap)
    
    # 3. 创建临时AnnData用于处理转换后的空间
    temp_adata = sc.AnnData(
        X=adata.obsm['X_latent_map'],  # 使用转换后的潜在空间作为X
        obs=adata.obs.copy(),
        layers={
            'velocity': adata.obsm['velocity_map_latent'].copy()  # 使用次空间速度
        }
    )
    
    # 4. 准备2D坐标（基于转换后的空间）
    if dim_reduction == 'umap':
        if 'X_umap' not in temp_adata.obsm:
            sc.pp.neighbors(temp_adata, use_rep='X')  # 基于转换后的空间计算邻居
            sc.tl.umap(temp_adata)
        temp_adata.obsm['X_umap'] = temp_adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in temp_adata.obsm:
            sc.pp.pca(temp_adata, n_comps=2, use_rep='X')  # 基于转换后的空间计算PCA
        temp_adata.obsm['X_umap'] = temp_adata.obsm['X_pca'][:, :2].copy()
    else:  # 'none'直接使用转换后潜在空间的前两维
        temp_adata.obsm['X_umap'] = temp_adata.X[:, :2].copy()
    
    # 5. 设置scVelo所需的层
    temp_adata.layers['Ms'] = temp_adata.X.copy()  # 模拟spliced层
    temp_adata.layers['velocity'] = temp_adata.layers['velocity'].copy()  # 次空间速度
    
    # 6. 处理2D速度投影
    if temp_adata.layers['velocity'].shape[1] == 2:
        # 速度已是2D，直接作为umap速度
        temp_adata.obsm['velocity_umap'] = temp_adata.layers['velocity']
    else:
        sc.pp.neighbors(temp_adata, n_neighbors=30, use_rep='X')  # 基于转换后的高维数据计算邻居
        scv.tl.velocity_graph(temp_adata, vkey='velocity', n_jobs=16)  # 构建速度图
        scv.tl.velocity_embedding(temp_adata, basis='umap', vkey='velocity')  # 投影到2D
    
    # 7. 定义势能网络（与原函数保持一致）
    class PotentialNet(nn.Module):
        def __init__(self, in_dim=2, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    # 8. 初始化并训练势能网络
    potential_net = PotentialNet(in_dim=temp_adata.obsm['X_umap'].shape[1])
    potential_net = potential_net.to(device)
    
    X = torch.tensor(temp_adata.obsm['X_umap'], dtype=torch.float32, device=device)
    V = torch.tensor(temp_adata.obsm['velocity_umap'], dtype=torch.float32, device=device)
    
    optimizer = optim.Adam(potential_net.parameters(), lr=1e-3)
    n_epochs = 2000
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        X.requires_grad = True
        phi = potential_net(X)
        grad_phi = torch.autograd.grad(
            phi, X, 
            grad_outputs=torch.ones_like(phi), 
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]  # 计算势能梯度
        pred_V = -grad_phi  # 速度是势能梯度的负值
        loss = ((pred_V - V)**2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, PotentialNet_loss: {loss.item():.6f}")
    
    # 9. 计算原始细胞的势能值
    umap_coords = temp_adata.obsm['X_umap']
    with torch.no_grad():
        input_tensor = torch.from_numpy(umap_coords).float().to(device)
        original_potentials = potential_net(input_tensor).cpu().numpy().flatten()
    
    # 10. 绘制势能景观图
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        x=umap_coords[::1, 0],
        y=umap_coords[::1, 1],
        hue=original_potentials[::1],
        palette='RdYlBu_r', 
        s=5,              
        alpha=0.8,           
        edgecolor='none',     
        ax=ax,
        legend=False   
    )
    
    # 添加颜色条
    norm = plt.Normalize(original_potentials.min(), original_potentials.max())
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Potential', fontsize=12)
    
    # 美化坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='box')
    
    # 保存图像
    if output_path is not None:
        plt.savefig(f"{output_path}/Potential_landscape_map.png", dpi=300, bbox_inches='tight')
        print(f"[plot_landscape_map] has been saved -> {output_path}/Potential_landscape_map.png")
    
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_interaction_potential_epoch(
    model,
    d=200,
    num_points=100,
    output_path=None,
    device="cuda"
):
    """
    Visualize interaction potential energy (input is distance d, output is energy value)
    
    Parameters:
        model: Trained model containing interaction network
        d: Range parameter for distance (d_min = -d, d_max = d)
        num_points: Number of sampling points
        output_path: Path to save the plot (do not save if None)
        device: Computing device
    """
    d_min = -d
    d_max = d
    # Load model
    model.to(device)
    model.eval()
    # Initialize ODE function (used for energy calculation)
    
    # Generate distance sequence
    d_values = np.linspace(d_min, d_max, num_points)
    d_tensor = torch.tensor(d_values, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension to match network input
    # Calculate energy (use no_grad to speed up and avoid gradient computation)
    with torch.no_grad():
        energies = model.interaction_net(d_tensor)
        # Critical fix: Move CUDA tensor to CPU and convert to NumPy array
        energies = energies.cpu().numpy().flatten()  # Flatten to 1D array to match d_values shape
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, energies, 'b-', linewidth=2)
    plt.xlabel('Distance (d)')
    plt.ylabel('Interaction Potential Energy')
    plt.title('Interaction Potential vs Distance')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        last_slash_index = output_path.rfind('/')
        new_path = output_path[:last_slash_index]
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print(f"Directory created: {new_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Interaction potential plot saved to: {output_path}")
    else:
        plt.show()
#%%
# Fit potential on 2D UMAP to create landscape
import torch.nn as nn
import torch.optim as optim
import torch

def plot_landscape(adata,model,output_path=None,dim_reduction="none",device="cuda"):

    # 1. Ensure velocity has been calculated
    if 'velocity_latent' not in adata.layers:
        adata = compute_velocity(adata, model,device)

    # 2. Prepare coordinates (2-D)
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2)
        adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].copy()
    else:  # 'none' directly uses first two dimensions of latent space
        adata.obsm['X_umap'] = adata.obsm['X_latent'][:, :2].copy()

    # 3. Place velocity in scVelo's specified key
    adata.layers['Ms']   = adata.obsm['X_latent'].copy()
    adata.layers['velocity'] = adata.obsm['velocity_latent']

    # 4. 2-D velocity processing
    if adata.layers['velocity'].shape[1] == 2:
        # Velocity is already 2-D, directly use as umap velocity
        adata.obsm['velocity_umap'] = adata.layers['velocity']
    else:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')  # Calculate neighbors based on high-dim data
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)  # Build velocity graph from high-dim velocity vectors
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')  # Project 

    if 'velocity_latent' not in adata.layers:
        adata = compute_velocity(adata, model,device)
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_latent')
            sc.tl.umap(adata)
        adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2, use_rep='X_latent')
        adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].copy()
    else:  # Directly use first two dimensions of latent space
        adata.obsm['X_umap'] = adata.obsm['X_latent'][:, :2].copy()


    X = torch.tensor(adata.obsm['X_umap'], dtype=torch.float32)
    V = torch.tensor(adata.obsm['velocity_umap'], dtype=torch.float32)

    class PotentialNet(nn.Module):
        def __init__(self, in_dim=2, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    potential_net = PotentialNet(in_dim=X.shape[1])
    potential_net = potential_net.cuda()
    X = X.cuda()
    V = V.cuda()

    optimizer = optim.Adam(potential_net.parameters(), lr=1e-3)
    n_epochs = 2000

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        X.requires_grad = True
        phi = potential_net(X)
        grad_phi = torch.autograd.grad(
            phi, X, 
            grad_outputs=torch.ones_like(phi), 
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]  # (N, 2)
        # Velocity is the negative gradient of potential
        pred_V = -grad_phi
        loss = ((pred_V - V)**2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, PotentialNet_loss: {loss.item():.6f}")

    # Calculating Potential for original cells
    umap_coords = adata.obsm['X_umap']
    with torch.no_grad():
        input_tensor = torch.from_numpy(umap_coords).float().to(device)
        original_potentials = potential_net(input_tensor).cpu().numpy().flatten()

    # Plot potential
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = sns.scatterplot(
        x=umap_coords[::1, 0],
        y=umap_coords[::1, 1],
        hue=original_potentials[::1],
        palette='RdYlBu_r', 
        s=5,              
        alpha=0.8,           
        edgecolor='none',     
        ax=ax,
        legend=False   
    )


    norm = plt.Normalize(original_potentials.min(), original_potentials.max())
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Potential', fontsize=12)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_aspect('equal', adjustable='box')
    plt.savefig(output_path+'/Potential_landscape.png', dpi=300, bbox_inches='tight')
    plt.grid(False)

    plt.tight_layout()
    plt.show()
#%%

import cellrank as cr



def analyze_terminal_states(adata, classified_type="cell_type", terminal_states=["type1", "type2"], output_path=None):
    """
    Analyze and plot terminal states, and compute fate probabilities.

    Parameters:
    adata: AnnData object containing single-cell data and velocity information.
    classified_type: String indicating the column in adata.obs that contains cell type classifications.
    terminal_states: List of strings representing the terminal states to be analyzed.
    output_path: String specifying the folder path where results will be saved.
    """

    # Check if classified_type exists in adata.obs
    if classified_type not in adata.obs.columns:
        raise ValueError(f"Error: '{classified_type}' does not exist in adata.obs. Please check the column name.")

    # Check if all terminal_states exist in classified_type
    missing_states = [state for state in terminal_states if state not in adata.obs[classified_type].unique()]
    if missing_states:
        raise ValueError(f"Error: The following terminal states do not exist in '{classified_type}': {missing_states}")

    # Ensure output folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if 'neighbors' not in adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep='X_latent')
    # Create VelocityKernel object and compute transition matrix
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    transition_matrix = vk.transition_matrix
    if not np.allclose(np.sum(transition_matrix, axis=1), 1, atol=1e-3):
        raise ValueError("Transition matrix rows do not sum to 1.")
    # Create GPCCA object
    g = cr.estimators.GPCCA(vk)

    # Manually define terminal states
    manual_terminal_states = {
        state: adata.obs_names[adata.obs[classified_type] == state].tolist()
        for state in terminal_states
    }

    # Set terminal states
    g.set_terminal_states(manual_terminal_states)

    # Plot terminal states
    g.plot_macrostates(which="terminal", mode = 'embedding', legend_loc="right margin",)

    print(f"Terminal states analysis is saved to {output_path}")
    g.compute_fate_probabilities(tol=1e-5, preconditioner="ilu")
    g.plot_fate_probabilities(mode="embedding", title='All fate probabilities', save=output_path + '/all_fate_probabilities_plot.svg')
#%%
from CytoBridge.tl.analysis import train_mlp_classifier,MLPClassifier
from anndata import AnnData
import joblib

def visualize_trajectory_classification(
    sde_point: np.ndarray,
    predicted_labels_list: list,
    output_path: str,
    adata: AnnData,
    time_points: Optional[np.ndarray] = None
):
    """
    Visualize cell type distribution at different time points
    
    Parameters:
        sde_point: SDE-generated trajectory points (n_time_points, n_cells, latent_dim)
        predicted_labels_list: List of predicted labels for each time point
        output_path: Path to save visualization results
        adata: Original data used to obtain time point information
        time_points: Time point array (optional)
    """
    os.makedirs(output_path, exist_ok=True)
    cmap = plt.cm.get_cmap('tab20')
    
    # Get all unique labels and map to colors
    all_labels = np.unique(np.concatenate(predicted_labels_list))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    # Get time points (extract from adata if not provided)
    if time_points is None:
        time_points = sorted(adata.obs['time_point_processed'].unique())
    
    # Plot for each time point
    for i in range(len(time_points)):
        t = time_points[i]
        # Extract SDE data for current time point
        data_t = sde_point[i]
        # Extract predicted labels
        labels_t = predicted_labels_list[i]
        
        # Convert labels to indices (for color mapping)
        label_indices = np.array([label_to_idx[label] for label in labels_t])
        colors = cmap(label_indices % cmap.N)
        
        # Plot (assuming data is already dimensionality-reduced, using first two dimensions)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data_t[:, 0], data_t[:, 1], c=colors, alpha=0.8, s=30)
        ax.set_title(f'Cell Type Distribution at Time {t}')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        
        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i % cmap.N), markersize=10) 
                   for i in range(len(all_labels))]
        ax.legend(handles, all_labels, title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'trajectory_time_{t}.png'), dpi=300)
        plt.show()
        plt.close()

def process_sde_classification(
    adata: AnnData,
    model: nn.Module,
    classifyed_type: str = 'cluster',
    sde_data_path: str = './sde_data',
    output_path: str = './classification_results',
    sigma: float = 0.05,
    n_time_steps: int = 10,
    sample_traj_num: int = 1000,
    train_mlp_classifier_epoches=400,
    init_time: Optional[int] = None,
    device: str = 'cuda',
    dim_reduction="none",
    hidden_size: int = 128  # Add hidden_size parameter to ensure consistency during loading and training
):
    """
    Complete workflow: Train classifier → Process SDE data → Classification prediction → Visualization
    
    Parameters:
        adata: AnnData object containing cell data
        model: Pretrained SDE model (used for generating trajectories)
        classifyed_type: Key of classification target in obs (default 'cluster')
        sde_data_path: Path to save SDE trajectory data
        output_path: Path to save results
        sigma: SDE noise parameter
        n_time_steps: Number of time steps
        sample_traj_num: Number of trajectories to sample
        init_time: Initial time point (uses the earliest time point in adata by default)
        device: Computing device
        hidden_size: MLP hidden layer dimension, used to maintain consistency when loading the model
    """
    model_path = os.path.join(output_path, 'mlp_classifier.pth')
    encoder_path = os.path.join(output_path, 'label_encoder.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        print("Loading existing MLP classifier...")
        label_encoder = joblib.load(encoder_path)
        input_size = adata.X.shape[1]
        num_classes = len(label_encoder.classes_)
        mlp_model = MLPClassifier(input_size, hidden_size, num_classes).to(device)
        mlp_model.load_state_dict(torch.load(model_path, map_location=device))
        mlp_model.eval()  
    else:
        print("Training MLP classifier...")
        mlp_model, label_encoder = train_mlp_classifier(
            adata,
            classifyed_type=classifyed_type,
            hidden_size=hidden_size,  
            device=device,
            train_mlp_classifier_epoches=train_mlp_classifier_epoches
        )
        # Save classifier
        os.makedirs(output_path, exist_ok=True)
        torch.save(mlp_model.state_dict(), model_path)
        joblib.dump(label_encoder, encoder_path)
    
    # 2. Process SDE data
    sde_point_path = os.path.join(sde_data_path, 'sde_point.npy')
    if not os.path.exists(sde_point_path):
        print("Generating SDE trajectories...")
        
        if init_time is None:
            init_time = sorted(adata.obs['time_point_processed'].unique())[0]
        # Generate SDE trajectories
        generate_sde_trajectories(
            adata=adata,
            exp_dir=sde_data_path,
            device=device,
            sigma=sigma,
            n_time_steps=n_time_steps,
            sample_traj_num=sample_traj_num,
            init_time=init_time
        )
    
    # Load SDE data
    print("Loading SDE data...")
    sde_point = np.load(sde_point_path)
    time_points = sorted(adata.obs['time_point_processed'].unique())
    assert len(sde_point) == len(time_points), "Number of time points in SDE data does not match adata"
    

    # 3. MLP classification prediction
    print("Predicting cell types for SDE trajectories...")
    mlp_model.eval()
    predicted_labels_list = []
    with torch.no_grad():
        for i in range(len(time_points)):
            # Get SDE data at current time point
            data_t = sde_point[i]
            data_t_tensor = torch.tensor(data_t, dtype=torch.float32).to(device)
            
            # Make predictions
            outputs = mlp_model(data_t_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.cpu().numpy())
            predicted_labels_list.append(predicted_labels)

    if (sde_point.shape[-1]==2):
        point_2d = sde_point[..., :2]  
    elif dim_reduction == 'pca':
        mean_ = adata.uns['pca']['mean'] if 'mean' in adata.uns['pca'] else 0.
        comps = adata.uns['pca']['components_'][:, :2]
        point_2d = np.dot(sde_point - mean_, comps)  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)
    elif dim_reduction == 'umap':
        reducer = UMAP().fit(adata.obsm['X_latent'])
        def flatten_and_2d(model, x):
            *lead, F = x.shape               
            x_2d = x.reshape(-1, F)         
            y_2d = model.transform(x_2d)    
            return y_2d.reshape(*lead, 2)   
        point_2d = flatten_and_2d(reducer, sde_point) # (n_time, n_traj, 2)
        
    else:  
        point_2d = sde_point[..., :2]  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)


    print("Visualizing results...")
    visualize_trajectory_classification(
        sde_point=point_2d,
        predicted_labels_list=predicted_labels_list,
        output_path=output_path,
        adata=adata,
        time_points=time_points
    )
    print(f"Results saved to {output_path}")

from CytoBridge.Map.tl.Mongemap import TransportMap

def plot_sde_trajectories_map(
    adata: sc.AnnData,
    tranmap: TransportMap,
    output_path: str = "./sde_results_map",
    device: str = "cuda",
    sigma: float = 1.0,
    n_bins: int = 10,
    n_trajectories: int = 100,
    init_time: int = 0,
    dim_reduction: str = 'none',
    split_true: bool = False
):
    os.makedirs(output_path, exist_ok=True)
    device = torch.device(device)
    print(f"Using device: {device}")

    # 生成 SDE 轨迹
    sde_traj, sde_point, w_point = generate_sde_trajectories(
        adata,
        exp_dir=output_path,
        device=device,
        sigma=sigma,
        n_time_steps=n_bins,
        sample_traj_num=n_trajectories,
        init_time=init_time,
        split_true=split_true
    )

    # 将轨迹数据和点数据转换为 torch.Tensor 并应用 TransportMap
    sde_traj = torch.tensor(sde_traj, dtype=torch.float32, device=device)
    sde_point = torch.tensor(sde_point, dtype=torch.float32, device=device)
    w_point = torch.tensor(w_point, dtype=torch.float32, device=device)

    # 应用 TransportMap 变换
    sde_traj = tranmap.T(sde_traj).cpu().numpy()
    sde_point = tranmap.T(sde_point).cpu().numpy()
    w_point = tranmap.T(w_point).cpu().numpy()

    # 降维到 2D 空间
    dim_reduction = str(dim_reduction).lower()
    if dim_reduction == 'umap' or dim_reduction == 'umap_adata':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        plot_2d = adata.obsm['X_umap'][:, :2]
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            raise ValueError('Please run sc.pp.pca(adata) first')
        plot_2d = adata.obsm['X_pca'][:, :2]
    elif dim_reduction in {'none', 'null', ''}:
        plot_2d = adata.obsm['X_latent'][:, :2]
    else:
        raise ValueError(f"Invalid dim_reduction: {dim_reduction}")

    time_key = 'time_point_processed'
    real_df = pd.DataFrame(plot_2d, columns=['x1', 'x2'])
    real_df['samples'] = adata.obs[time_key].values

    # 降维轨迹数据
    if sde_traj.shape[-1] > 2:
        if dim_reduction == 'pca':
            mean_ = adata.uns['pca']['mean'] if 'mean' in adata.uns['pca'] else 0.
            comps = adata.uns['pca']['components_'][:, :2]
            traj_2d = np.dot(sde_traj - mean_, comps)
            point_2d = np.dot(sde_point - mean_, comps)
        elif dim_reduction == 'umap':
            umap_path = os.path.join(output_path, 'umap_model.pkl')
            reducer = get_or_train_umap(
                data=adata.obsm['X_latent'],
                save_path=umap_path,
                n_neighbors=10,
                min_dist=0.2,
                n_components=2,
                random_state=42
            )
            traj_2d = apply_umap_transform(reducer, sde_traj)
            point_2d = apply_umap_transform(reducer, sde_point)
        elif dim_reduction == 'umap_adata':
            # 获取轨迹和点的形状信息
            T_bins, M, _ = sde_traj.shape
            T, _, _ = sde_point.shape

            # 调整轨迹和点的形状为二维
            traj_reshaped = sde_traj.reshape(-1, sde_traj.shape[-1])  # (T_bins*M, D)
            point_reshaped = sde_point.reshape(-1, sde_point.shape[-1])  # (T*M, D)

            # 创建临时 adata，包含所有需要降维的数据
            temp_adata = sc.AnnData(
                X=np.concatenate([adata.obsm['X_latent'], traj_reshaped, point_reshaped], axis=0)
            )
            # 存储原始数据的索引边界
            x_raw_end = len(adata.obsm['X_latent'])
            traj_end = x_raw_end + len(traj_reshaped)

            # 使用 X_latent 计算邻居并进行 UMAP 降维
            sc.pp.neighbors(temp_adata)
            sc.tl.umap(temp_adata)

            # 提取降维后的结果
            X_2d = temp_adata.obsm['X_umap'][:x_raw_end]  # (batchsize, 2)
            traj_2d_unshape = temp_adata.obsm['X_umap'][x_raw_end:traj_end]  # (T_bins*M, 2)
            point_2d_unshape = temp_adata.obsm['X_umap'][traj_end:]  # (T*M, 2)

            # 恢复原始形状
            traj_2d = traj_2d_unshape.reshape(T_bins, M, 2)  # (T_bins, M, 2)
            point_2d = point_2d_unshape.reshape(T, M, 2)  # (T, M, 2)
        else:
            raise ValueError(f"Invalid dim_reduction: {dim_reduction}")
    else:
        traj_2d = sde_traj
        point_2d = sde_point

    print("Plotting SDE trajectories with TransportMap transformation...")
    sde_plot(df=real_df,  
             generated=point_2d,
             trajectories=traj_2d,
             save=True,  
             output_path=output_path,  
             file='sde_trajectories_map.pdf')

    return adata
