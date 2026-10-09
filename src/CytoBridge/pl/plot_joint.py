import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import os
import pickle
import sys
import umap
import numpy as np
from typing import Optional, Tuple
import torch
from .plot_ode_v3 import plot_ode_v3 as plot_ode_v3_core

# Load UMAP estimators archived with the pynndescent/numba versions used for
# the manuscript, without refitting their learned embedding.
try:  # pragma: no cover - exercised by legacy paper assets
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



#%%
from CytoBridge.tl.analysis import compute_map_X_latent

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from CytoBridge.tl.analysis import generate_ode_trajectories
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
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

#%%
import numpy as np
import os
from typing import Tuple
try:
    from mofapy2.run.entry_point import entry_point
except ImportError:  # MOFA plots are optional; core UMAP helpers remain usable
    entry_point = None

import numpy as np
import os
import pickle
from typing import Tuple, Dict
try:
    from mofapy2.run.entry_point import entry_point
except ImportError:  # MOFA plots are optional; core UMAP helpers remain usable
    entry_point = None

def train_and_save_mofa_model_simple(
    data_matrices: list,
    params_path: str,
    n_factors: int = 10,
    random_state: int = 42,
    max_iter: int = 1000
) -> Dict[str, np.ndarray]:
    """
    训练MOFA并保存权重矩阵W
    """
    if entry_point is None:
        raise ImportError("mofapy2 is required for MOFA model training")
    print("  构建MOFA模型...")
    ent = entry_point()
    ent.set_data_matrix(data_matrices)
    ent.set_model_options(factors=n_factors)
    ent.set_train_options(iter=max_iter, seed=random_state, verbose=False)
    ent.build()
    ent.run()
    
    # 提取权重矩阵W
    # W_list: [W_view1, W_view2], 每个W: (n_features, n_factors)
    W_list = ent.model.nodes["W"].getExpectation()
    
    # 保存
    model_params = {'W_list': W_list, 'n_factors': n_factors}
    
    with open(params_path, 'wb') as f:
        pickle.dump(model_params, f)
    
    print(f"  模型保存: {params_path}")
    print(f"  W矩阵形状: {[W.shape for W in W_list]}")
    
    return model_params


def mofa_transform_direct(
    X_list: list,
    model_params: Dict[str, np.ndarray],
    return_2d: bool = True,
    average_mode: str = "all"   # 新增参数
) -> np.ndarray:
    """
    直接投影降维：Z = X @ W
    
    参数
    ----
    X_list: [X_view1, X_view2]
        每个X: (n_samples, n_features)
    model_params: dict
        包含W_list
    return_2d: bool
        是否只返回前2个因子
    average_mode: str
        平均方式：
        "all"   – 全部view平均（默认）
        "first" – 仅用第1个view
        "second"  – 仅用第2个view
        "12"    – 仅第1、2个view平均（等价于all，当view=2时）
        
    返回
    ----
    Z: (n_samples, n_factors) or (n_samples, 2)
    """
    W_list = model_params['W_list']
    
    # 对每个view计算: X @ W
    Z_from_views = [X @ W for X, W in zip(X_list, W_list)]
    
    # 根据模式选择要平均的view
    if average_mode == "all":
        selected = Z_from_views
    elif average_mode == "first":
        selected = Z_from_views[:1]
    elif average_mode == "second":
        selected = Z_from_views[1:2]
    elif average_mode == "12":
        selected = Z_from_views[:2]
    else:
        raise ValueError("unsupported average_mode: {}".format(average_mode))
    
    # 平均选中的view
    Z_avg = np.mean(selected, axis=0)
    
    print(f"    各view结果: {[Z.shape for Z in Z_from_views]}")
    print(f"    选中平均的view: {len(selected)} 个")
    print(f"    平均后: {Z_avg.shape}")
    print(f"    Z范围: [{Z_avg.min():.3f}, {Z_avg.max():.3f}]")
    
    if return_2d:
        return Z_avg[:, :2]
    return Z_avg


def load_mofa_params(params_path: str) -> Dict[str, np.ndarray]:
    """
    从pickle文件加载MOFA参数
    
    这是关键点：不依赖mofax或HDF5，只用自己的pickle文件
    """    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"未找到MOFA模型文件: {params_path}")
    
    with open(params_path, 'rb') as f:
        model_params = pickle.load(f)
    
    print(f"  加载模型参数: {params_path}")
    print(f"  因子数量: {model_params['n_factors']}")
    print(f"  W矩阵形状: {[w.shape for w in model_params['W_list']]}")
    
    return model_params

def process_ode_3d_simple(
    mainpath: str,
    tranmap,
    data_type: str,
    model_params: Dict[str, np.ndarray],
    transform_func,
    average_mode: str = "all",
    T_direction: str = "ori",
    device="cuda"
) -> np.ndarray:
    """
    使用简单投影处理ODE的3D数组
    """
    path = os.path.join(mainpath, f"ode_results/{data_type}.npy")
    
    array = np.load(path)
    t_dim, n_trajectories, D = array.shape
    array = array.reshape(-1, D)

    array_tensor = torch.tensor(array, dtype=torch.float32, device=device)
    print("array.min()",array.min(),"array.max()",array.max())
    if  T_direction == "ori":
        array_map = tranmap.T(array_tensor).cpu().numpy()
        print("array_map.min()",array_map.min(),"array_map.max()",array_map.max())

    elif T_direction == "rev":
        array_map = tranmap.T_rev(array_tensor).cpu().numpy()

    print(f"    处理 {data_type}:")
    print(f"      原始: {array.shape} + {array_map.shape}")
    
    print(T_direction)
    # 降维
    if T_direction =="rev":
        Z_2d = transform_func([array_map, array], model_params, return_2d=True,average_mode=average_mode)
        result = Z_2d.reshape(t_dim, n_trajectories, 2)
    elif T_direction =="ori":
        Z_2d = transform_func([array, array_map], model_params, return_2d=True,average_mode=average_mode)
        result = Z_2d.reshape(t_dim, n_trajectories, 2)

    # Reshape回3D
    print(f"      结果: {result.shape}")
    
    return result


def plot_ode(X, point_array, traj_array, save_dir):
    """
    Backward-compatible wrapper that routes joint visualizations to plot_ode_v3.
    """
    del point_array  # kept for compatibility with old callers
    n_time = max(len(X), 1)
    time_ticks = np.arange(n_time, dtype=float)
    if n_time == 1:
        traj_times = np.zeros(traj_array.shape[0], dtype=float)
    else:
        traj_times = np.linspace(float(time_ticks[0]), float(time_ticks[-1]), traj_array.shape[0])

    plot_ode_v3_core(
        X=X,
        traj_array=traj_array,
        save_path=save_dir,
        traj_times=traj_times,
        time_ticks=time_ticks,
        traj_cmap="sunset",
        background_mode="auto",
        show_axes=True,
    )


def plot_joint_mofa(
    adata,
    tranmap,
    adata_path: str,
    generate_path: str,
    device: str = "cuda",
    n_factors: int = 5,
    random_state: int = 42,
    force_retrain: bool = False,
    max_iter: int = 1000,
    T_direction ="ori",
    average_mode = "all",
    average_mode_X= "all",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    最终简化版：直接投影降维
    #1个数据的原始数据+映射map数据作为学习数据

    流程:
    1. 训练MOFA得到W
    2. Z = X @ W (直接矩阵乘法)
    3. 完成
    """
    
    print("="*60)
    print("MOFA直接投影降维")
    print("="*60)
    
    # 1. 准备数据
    X_raw = adata.obsm['X_latent']
    if 'X_latent_map' not in adata.obsm:
        adata = compute_map_X_latent(adata, tranmap, device)
    X_map_raw = adata.obsm['X_latent_map']
    
    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())
    X_raw_list = [X_raw[adata.obs[samples_key] == t] for t in unique_times]
    X_map_list = [X_map_raw[adata.obs[samples_key] == t] for t in unique_times]
    
    X_raw_combined = np.concatenate(X_raw_list, axis=0)
    X_map_combined = np.concatenate(X_map_list, axis=0)
    
    train_data = [[X_raw_combined], [X_map_combined]]
    
    # 2. 训练或加载MOFA模型
    save_path = os.path.join(adata_path, "joint")
    os.makedirs(save_path, exist_ok=True)
    
    params_path = os.path.join(save_path, "mofa_model_params.pkl")
    
    if os.path.exists(params_path) and not force_retrain:
        print("步骤2: 加载模型...")
        model_params = load_mofa_params(params_path)
    else:
        print("步骤2: 训练新模型...")
        model_params = train_and_save_mofa_model_simple(
            train_data, params_path, n_factors, random_state, max_iter
        )
    
    # 3. adata降维
    print("步骤3: adata降维...")
    adata_Z_list = []
    for i, time_point in enumerate(unique_times):
        X_t_raw = X_raw_list[i]
        X_t_map = X_map_list[i]
        
        Z_t = mofa_transform_direct([X_t_raw, X_t_map], model_params,average_mode=average_mode_X)
        adata_Z_list.append(Z_t)
        print(f"  时间点 {time_point}: {X_t_raw.shape} -> {Z_t.shape}")
    
    # 4. ODE point降维
    print("步骤4: ODE point降维...")
    if T_direction== "ori":
        generate_path=adata_path
        print("generate_path :",generate_path)
    point_2d = process_ode_3d_simple(generate_path,tranmap, "ode_point", model_params, mofa_transform_direct,average_mode,T_direction,device)
    
    # 5. ODE traj降维
    print("步骤5: ODE traj降维...")
    traj_2d = process_ode_3d_simple(generate_path,tranmap, "ode_traj", model_params, mofa_transform_direct,average_mode,T_direction,device)


    # 6. 可视化
    print("步骤6: 绘图...")
    save_dir = os.path.join(save_path, f'map_joint_mofa_trajectories_{T_direction}_{average_mode_X}_{average_mode}.pdf')
    plot_ode(X=adata_Z_list, point_array=point_2d, traj_array=traj_2d, save_dir=save_dir)

    print("="*60)
    print(f"✅ 完成！结果: {save_dir}")
    print("="*60)
    
    adata_2d_combined = np.concatenate(adata_Z_list, axis=0)
    return point_2d, traj_2d, adata_2d_combined

def plot_joint_mofa_true(
    adata,
    adata_map,
    tranmap,
    adata_path: str,
    generate_path: str,
    device: str = "cuda",
    n_factors: int = 5,
    random_state: int = 42,
    force_retrain: bool = False,
    max_iter: int = 1000,
    T_direction ="ori",
    average_mode_X = "all",
    average_mode = "all",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    最终简化版：直接投影降维
    #两个数据的原始数据作为学习数据
    
    流程:
    1. 训练MOFA得到W
    2. Z = X @ W (直接矩阵乘法)
    3. 完成
    """
    
    print("="*60)
    print("MOFA直接投影降维")
    print("="*60)
    
    # 1. 准备数据
    X_raw = adata.obsm['X_latent']
    print("X_raw.min()",X_raw.min(),"X_raw.max()",X_raw.max())
    X_map_raw = adata_map.obsm['X_latent']
    print("X_map_raw.min()",X_map_raw.min(),"X_map_raw.max()",X_map_raw.max())

    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())
    X_raw_list = [X_raw[adata.obs[samples_key] == t] for t in unique_times]
    X_map_list = [X_map_raw[adata_map.obs[samples_key] == t] for t in unique_times]
    
    X_raw_combined = np.concatenate(X_raw_list, axis=0)
    X_map_combined = np.concatenate(X_map_list, axis=0)
    
    train_data = [[X_raw_combined], [X_map_combined]]
    
    # 2. 训练或加载MOFA模型
    save_path = os.path.join(adata_path, "joint")
    os.makedirs(save_path, exist_ok=True)
    
    params_path = os.path.join(save_path, "true_mofa_model_params.pkl")
    
    if os.path.exists(params_path) and not force_retrain:
        print("步骤2: 加载模型...")
        model_params = load_mofa_params(params_path)
    else:
        print("步骤2: 训练新模型...")
        model_params = train_and_save_mofa_model_simple(
            train_data, params_path, n_factors, random_state, max_iter
        )
    
    # 3. adata降维
    print("步骤3: adata降维...")
    adata_Z_list = []
    for i, time_point in enumerate(unique_times):
        X_t_raw = X_raw_list[i]
        X_t_map = X_map_list[i]
        
        Z_t = mofa_transform_direct([X_t_raw, X_t_map], model_params,average_mode=average_mode_X)
        adata_Z_list.append(Z_t)
        print(f"  时间点 {time_point}: {X_t_raw.shape} -> {Z_t.shape}")
    
    # 4. ODE point降维
    print("步骤4: ODE point降维...")
    if T_direction== "ori":
        generate_path=adata_path
        print("generate_path :",generate_path)
    point_2d = process_ode_3d_simple(generate_path,tranmap, "ode_point", model_params, mofa_transform_direct,average_mode,T_direction,device)
    
    # 5. ODE traj降维
    print("步骤5: ODE traj降维...")
    traj_2d = process_ode_3d_simple(generate_path,tranmap, "ode_traj", model_params, mofa_transform_direct,average_mode,T_direction,device)


    # 6. 可视化
    print("步骤6: 绘图...")
    save_dir = os.path.join(save_path, f'Ture_joint_mofa_trajectories_{T_direction}_{average_mode_X}_{average_mode}.pdf')
    plot_ode(X=adata_Z_list, point_array=point_2d, traj_array=traj_2d, save_dir=save_dir)
    
    print("="*60)
    print(f"✅ 完成！结果: {save_path}")
    print("="*60)
    
    adata_2d_combined = np.concatenate(adata_Z_list, axis=0)
    return point_2d, traj_2d, adata_2d_combined
#%%

def plot_joint_ode_GAUDI(adata,tranmap,mainpath: str,device="cuda",n_neighbors: int = 10 ,min_dist: float = 0.2,random_state: int = 42,force_retrain: bool = False):
    """
    读取 ode_point/ode_traj 及其 map 版本，合并后统一用 UMAP 降维到 2D。

    参数
    ----
    mainpath : str
        实验根目录，例如 /home/.../experiment_gse253582_21_ea0.1_eo0.05/
    n_neighbors, min_dist, random_state : int/float
        UMAP 参数
    force_retrain : bool
        是否强制重新训练 UMAP

    返回
    ----
    point_joint_2d : ndarray, shape (n_points, n_trajectories, 2)
    tra_joint_2d : ndarray, shape (n_bins, n_trajectories, 2)
    """
    X_raw = adata.obsm['X_latent']

    if 'X_latent_map' not in adata.obsm:
        adata = compute_map_X_latent(adata, tranmap,device)
    X_map_raw = adata.obsm['X_latent_map']

    point_X = np.concatenate(
        [X_raw, X_map_raw], axis=1
    )

    os.makedirs(os.path.join(mainpath, "joint"), exist_ok=True)
    save_path = os.path.join(mainpath, "joint/")
    umap_save_path = os.path.join(save_path, "umap_model_joint.pkl")
    reducer = get_or_train_umap(
        data=point_X,
        save_path=umap_save_path,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        force_retrain=force_retrain
    )


    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())    
    X_2d_joint_unshape = reducer.transform(point_X) 
    X_2d_joint= [X_2d_joint_unshape[adata.obs[samples_key] == t] for t in unique_times]


    # ---- 1. ode_point ----
    ode_point_path = os.path.join(mainpath, "ode_results/ode_point.npy")
    point_array = np.load(ode_point_path)

    n_points, n_trajectories1, D = point_array.shape
    point_array = point_array.reshape(-1, D)



    array_tensor = torch.tensor(point_array, dtype=torch.float32, device=device)
    point_array_map = tranmap.T(array_tensor).cpu().numpy()


    point_joint = np.concatenate(
        [point_array, point_array_map], axis=1
    )  # (n_points*n_trajectories, D1+D2)
    point_joint_2d = apply_umap_transform(reducer, point_joint)

    # ---- 2. ode_traj ----
    ode_traj_path = os.path.join(mainpath, "ode_results/ode_traj.npy")
    traj_array = np.load(ode_traj_path)

    n_trajs, n_trajectories2, D = traj_array.shape
    traj_array = traj_array.reshape(-1, D)

    array_tensor = torch.tensor(traj_array, dtype=torch.float32, device=device)
    traj_array_map = tranmap.T(array_tensor).cpu().numpy()

    traj_joint = np.concatenate(
        [traj_array, traj_array_map], axis=1
    )
    traj_joint_2d = apply_umap_transform(reducer, traj_array)
    
    shaped_point_joint_2d = point_joint_2d.reshape(n_points, n_trajectories1, 2)
    shaped_traj_joint_2d = traj_joint_2d.reshape(n_trajs, n_trajectories2, 2)

    save_base_name='joint_ode_trajectories_GAUDI.pdf'
    save_dir = os.path.join(save_path, save_base_name)
    
    plot_ode(X=X_2d_joint,
             point_array=shaped_point_joint_2d,
             traj_array=shaped_traj_joint_2d,
             save_dir=save_dir)
