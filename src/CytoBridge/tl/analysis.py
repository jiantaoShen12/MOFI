from CytoBridge.utils import load_model_from_adata,check_submodules
import torch
import math
import numpy as np

def compute_map_X_latent(adata, tranmap,device="cuda"):
    """
    基于传输映射计算转换后的潜在空间并存储到adata.obsm['X_latent_map']
    
    参数:
        adata: AnnData对象，包含'obsm['X_latent']'
        tranmap: TransportMap实例，用于执行空间转换
    """
    if 'X_latent' not in adata.obsm:
        raise ValueError("adata.obsm中未找到'X_latent'，请先计算潜在空间")
    
    # compute_map_X_latent
    x_latent = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    x_latent_map = tranmap.T(x_latent).cpu().numpy()
    
    # save map results
    adata.obsm['X_latent_map'] = x_latent_map
    print(f"[compute_map_X_latent] has been computed and saved in adata.obsm['X_latent_map']")
    return adata

def compute_velocity(adata, model, device='cuda'):
    '''
    Calculate velocity based on the trained model 
    '''
    model.to(device)
    # Calculate velocity and growth on the dataset for downstream analysis
    all_times = torch.tensor(adata.obs['time_point_processed']).unsqueeze(1).float().to(device)
    all_data = torch.tensor(adata.obsm['X_latent']).float().to(device)
    net_input = torch.cat([all_data, all_times], dim = 1)
    with torch.no_grad():
        velocity = model.velocity_net(net_input)
    # Store in obsm instead of layers to avoid dimension mismatch
    # layers requires shape (n_obs, n_vars) but velocity is (n_obs, latent_dim)
    adata.obsm['velocity_latent'] = velocity.detach().cpu().numpy()

    return adata

def compute_velocity_map(adata, model,tranmap, device='cuda'):
    
    '''
    Calculate velocity based on the trained model 
    '''

    if 'velocity_latent' not in adata.obsm:
        adata = compute_velocity(adata, model,device)
    model.to(device)
    # Calculate velocity and growth on the dataset for downstream analysis
    all_data = torch.tensor(adata.obsm['X_latent']).float().to(device)
    velocity_map =   tranmap.sec_vecolity(all_data,adata.obsm['velocity_latent'])#x,v
    adata.obsm['velocity_map_latent'] = velocity_map.detach().cpu().numpy()

    return adata

#%%
from CytoBridge.utils import load_model_from_adata
import torch
import numpy as np
from CytoBridge.tl.interaction import cal_interaction
def compute_interaction_force(adata, device='cuda'):
    '''
    Calculate interaction force based on the trained model
    '''
    model = load_model_from_adata(adata)
    model.to(device)
    
    # Check if interaction component exists in the model
    if 'interaction' not in model.components:
        raise ValueError("Model does not contain interaction component. Please ensure the model was trained with interaction.")
    
    # Prepare input data: latent features + time
    all_times = torch.tensor(adata.obs['time_point_processed']).unsqueeze(1).float().to(device)
    all_data = torch.tensor(adata.obsm['X_latent']).float().to(device)
    
    # Initialize log weights (required for interaction calculation)
    print("interaction_force use use_mass,",model.use_growth_in_ode_inter)
    if model.use_growth_in_ode_inter:
        if 'growth_rate' not in adata.obsm.keys():
            raise ValueError("Growth rate not found in adata.obsm. Please check if the growth term is set or the model is trained.")
        lnw = torch.tensor(adata.obsm['growth_rate']).float().to(device)

    interaction_force = cal_interaction(
        z=all_data,
        lnw=lnw,
        interaction_potential=model.interaction_net,
        cutoff=model.interaction_net.cutoff,
        use_mass=model.use_growth_in_ode_inter
    ).float()
    
    adata.layers['interaction_force'] = interaction_force.detach().cpu().numpy()
    
    # Store interaction force in adata layers
    adata.layers['interaction_force'] = interaction_force.detach().cpu().numpy()
    
    return adata
#%%

import os
import numpy as np
import torch
from typing import Tuple, Union

# If CytoSDE / euler_sdeint / load_model_from_adata are defined in other files,
# import them here according to the actual file structure
# from .model import CytoSDE, load_model_from_adata
# from .integrator import euler_sdeint

# ------------------------------------------------------------------------------
# SDE Definition Dependencies: PyTorch Neural Network Module
import torch
from torch.nn import Module


class CytoSDE(Module):
    """
    SDE definition for CytoBridge models (Ito-type with diagonal noise).
    Computes drift (f) and diffusion (g) terms for latent state (z) and log-weight (lnw).
    
    Attributes
    ----------
    noise_type : str
        Type of noise ("diagonal" = independent noise per dimension).
    sde_type : str
        Type of SDE ("ito" = Ito calculus).
    model : torch.nn.Module
        Pre-trained CytoBridge model (must include velocity/score/growth networks as components).
    sigma : float
        Global diffusion coefficient that scales the magnitude of noise.
    """
    noise_type = "diagonal"  # Diagonal noise (independent across dimensions)
    sde_type = "ito"         # Ito-type SDE

    def __init__(self, model, sigma=1.0):
        super().__init__()
        self.model = model    # Trained CytoBridge model (contains velocity/score/growth networks)
        self.sigma = sigma    # Diffusion coefficient for noise scaling

    def f(self, t, y):
        """
        Compute drift term f(t, y) for the SDE.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (shape: [1]).
        y : Tuple[torch.Tensor, torch.Tensor]
            Current state (z, lnw), where:
            - z: Latent state (shape: [batch_size, latent_dim])
            - lnw: Log-weight for particles (shape: [batch_size, 1])
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Drift terms for z and lnw (same shapes as input y).
        """
        z, lnw = y
        # Expand time to match the batch size of z
        t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
        # Input to velocity network: latent features + time
        vel_in = torch.cat([z, t_expand], 1)
        
        # Compute drift for z (base velocity + score correction if available)
        drift_z = self.model.velocity_net(vel_in)
        if "score" in self.model.components:
            _, gradients = self.model.compute_score(t, z)  # Assume compute_score returns (score, gradients)
            drift_z += gradients
        
        # Compute drift for lnw (growth term if available, else zero)
        if "growth" in self.model.components:
            drift_lnw = self.model.growth_net(vel_in)
        else:
            drift_lnw = torch.zeros_like(lnw)
        if "interaction" in self.model.components:
            interaction_force = cal_interaction(z=z,lnw=lnw,interaction_potential=self.model.interaction_net,cutoff=self.model.interaction_net.cutoff,use_mass=self.model.use_growth_in_ode_inter)
            drift_z += interaction_force
        return (drift_z, drift_lnw)

    def g(self, t, y):
        """
        Compute diffusion term g(t, y) for the SDE.
        Noise is applied to z (latent state) but not to lnw (log-weight).
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (shape: [1]).
        y : Tuple[torch.Tensor, torch.Tensor]
            Current state (z, lnw) (shapes: [batch_size, latent_dim], [batch_size, 1]).
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Diffusion terms for z and lnw (same shapes as input y).
        """
        z, lnw = y
        return (torch.ones_like(z) * self.sigma,  # Diffusion for z
                torch.ones_like(lnw) * self.sigma * 0.0)  # No diffusion for lnw


# ------------------------------------------------------------------------------
# Euler Integration Dependencies: Basic Math, PyTorch Tensor Operations
import math
import torch

def euler_sdeint_split(sde, initial_state, dt, ts, noise_std = 0.01):
    device = initial_state[0].device
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0
    output_states = []
    ts_list = ts.tolist()
    next_output_idx = 0
    w_prev = torch.exp(current_state[1])
    while current_time <= tf + 1e-8:
        t_tensor = torch.tensor([current_time], dtype=torch.float32, device=device)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z ,g_lnw = sde.g(t_tensor, current_state)
        noise_lnw = torch.randn_like(current_state[1]) * math.sqrt(dt)
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt+ g_lnw * noise_lnw

        current_time += dt
        if current_time >= ts_List[next_output_idx] - 1e-8:
            w_next = torch.exp(new_lnw)
            r = w_next / w_prev
            next_z = []
            next_lnw = []
            for j in range(current_state[0].shape[0]):
                if r[j] >= 1:
                    r_floor = torch.floor(r[j])
                    m_j = int(r_floor) + (1 if torch.rand(1, device=device) < (r[j] - r_floor) else 0)
                    for _ in range(m_j):
                        noise = torch.normal(0, noise_std, size=new_z[j].shape, device=device)
                        perturbed_x = new_z[j] + noise
                        next_z.append(perturbed_x.unsqueeze(0))
                        next_lnw.append(new_lnw[j].unsqueeze(0))
                else:
                    if torch.rand(1, device=device) < r[j]:
                        next_z.append(new_z[j].unsqueeze(0))
                        next_lnw.append(new_lnw[j].unsqueeze(0))
            if next_z:
                new_z = torch.cat(next_z, dim=0)
                new_lnw = torch.log(torch.ones(new_z.shape[0], 1) / initial_state[0].shape[0]).to(device)
            else:
                new_z = torch.empty(0, current_state[0].shape[1], device=device)
                new_lnw = torch.empty(0, 1, device=device)
            current_state = (new_z, new_lnw)
            output_states.append(current_state)
            next_output_idx += 1
            w_prev = torch.exp(new_lnw)
            if next_output_idx >= len(ts_list):
                break
        else:
            current_state = (new_z, new_lnw)
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    traj_z = [state[0] for state in output_states]
    traj_lnw = [state[1] for state in output_states]
    return traj_z, traj_lnw

def euler_sdeint(sde, y0, dt, ts):
    """
    Numerical integration of SDE using Euler-Maruyama method (for Ito-type SDEs).
    
    Parameters
    ----------
    sde : CytoSDE
        SDE object with f (drift) and g (diffusion) methods.
    y0 : Tuple[torch.Tensor, torch.Tensor]
        Initial state (z0, lnw0):
        - z0: Initial latent state (shape: [batch_size, latent_dim])
        - lnw0: Initial log-weight (shape: [batch_size, 1])
    dt : float
        Integration time step (fixed).
    ts : torch.Tensor
        Target time points to record (shape: [n_time_points], sorted in ascending order).
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - z_traj: Latent state trajectories (shape: [n_time_points, batch_size, latent_dim])
        - lnw_traj: Log-weight trajectories (shape: [n_time_points, batch_size, 1])
    """
    # Initialize state and current time
    y, t = y0, ts[0].item()
    out = []  # Store states at target time points
    ts_lst = ts.tolist()  # Convert target times to list for iteration
    idx = 0  # Index for target time points

    # Iterate until exceeding the maximum target time
    while t <= ts_lst[-1] + 1e-8:
        # Record state if current time matches target time (within tolerance)
        if t >= ts_lst[idx] - 1e-8:
            out.append(y)
            idx += 1
            if idx >= len(ts_lst):
                break  # Exit if all target times are recorded
        
        # Convert current time to tensor (matches model input format)
        t_tensor = torch.tensor([t], dtype=torch.float32, device=y[0].device)
        
        # Compute drift and diffusion terms
        f_z, f_lnw = sde.f(t_tensor, y)
        g_z, g_lnw = sde.g(t_tensor, y)
        
        # Generate Gaussian noise (scaled by sqrt(dt) for Ito calculus)
        noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
        noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
        
        # Euler-Maruyama update: y_{t+dt} = y_t + f*dt + g*noise
        y = (y[0] + f_z * dt + g_z * noise_z,
             y[1] + f_lnw * dt + g_lnw * noise_lnw)
        
        # Step forward in time
        t += dt

    # Fill missing target times with the last recorded state (due to numerical tolerance)
    while len(out) < len(ts_lst):
        out.append(out[-1])

    # Reshape output to [n_time_points, batch_size, ...]
    z_traj = torch.stack([o[0] for o in out])
    lnw_traj = torch.stack([o[1] for o in out])
    return z_traj, lnw_traj

def generate_sde_trajectories(
    adata,
    exp_dir: str,
    device: Union[str, torch.device],
    sigma: float,
    n_time_steps: int,
    sample_traj_num: int,
    init_time: int,
    split_true: bool =False

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encapsulates the complete workflow from the parent function (model loading → SDE integration → sampling/filtering → saving) into a standalone function.

    Returns
    -------
    sde_traj : np.ndarray
        Shape (n_steps, n_sample_cells, latent_dim)
    sde_point : np.ndarray
        Shape (n_time_points, n_cells_used, latent_dim)
    w_point : np.ndarray
        Shape (n_time_points, n_cells_used, 1) - Normalized weights
    """
    device = torch.device(device)
    os.makedirs(exp_dir, exist_ok=True)
    # ---------- Load Pre-trained Model ----------
    print("Loading model from adata...")
    model = load_model_from_adata(adata).to(device).float()  # Assume load_model_from_adata is predefined
    model.eval()
    print(f"Model components: {model.components}")

    # ---------- 2. Time Axis ----------
    time_key = "time_point_processed"
    all_times = sorted(adata.obs[time_key].unique())
    ts_points = torch.tensor(all_times, dtype=torch.float32, device=device)
    ts_continuous = torch.linspace(
        all_times[0], all_times[-1], n_time_steps,
        dtype=torch.float32, device=device
    )

    # ---------- 3. Initial State ----------
    init_mask = adata.obs[time_key] == init_time
    if not init_mask.any():
        raise ValueError(f"No data found for init_time={init_time}")
    x0 = torch.tensor(adata[init_mask].obsm["X_latent"], dtype=torch.float32, device=device)
    x0 = x0.requires_grad_(True)

    length = x0.shape[0]
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / length)
    initial_state = (x0, lnw0)

    # ---------- 4. SDE Integration ----------
    sde = CytoSDE(model, sigma).to(device)
    dt = (all_times[-1] - all_times[0]) / (n_time_steps - 1)
    if split_true :
        sde_traj, traj_lnw = euler_sdeint_split(sde, initial_state, dt, ts_continuous)
        sde_point, point_lnw = euler_sdeint_split(sde, initial_state, dt, ts_points)
    else :
        sde_traj, traj_lnw = euler_sdeint(sde, initial_state, dt, ts_continuous)
        sde_point, point_lnw = euler_sdeint(sde, initial_state, dt, ts_points)
    # ---------- 5. Sampling / Consistency Filtering ----------
    sample_traj_num = min(sample_traj_num, x0.shape[0])
    traj_cell_indices = torch.randperm(x0.shape[0], device=device)[:sample_traj_num]


    sde_traj = sde_traj[:, traj_cell_indices]
    traj_lnw = traj_lnw[:, traj_cell_indices]


    # ---------- 6. Convert to NumPy / Weight Normalization ----------
    sde_traj = sde_traj.detach().cpu().numpy()
    sde_point = sde_point.detach().cpu().numpy()
    w_point = torch.exp(point_lnw).detach().cpu().numpy()
    w_point /= w_point.sum(axis=1, keepdims=True)

    # ---------- 7. Saving ----------
    np.save(os.path.join(exp_dir, "sde_trajec.npy"), sde_traj)
    np.save(os.path.join(exp_dir, "sde_point.npy"), sde_point)
    np.save(os.path.join(exp_dir, "sde_weight.npy"), w_point)

    return sde_traj, sde_point, w_point

#%%
import torch
import math
import numpy as np
from torchdiffeq import odeint

def euler_odeint_split(ode_func, initial_state, ts, noise_std=0.01):
    device = initial_state[0].device
    t0, tf = ts[0].item(), ts[-1].item()
    current_t = t0
    current_z, current_lnw, current_m = initial_state
    w_prev = torch.exp(current_lnw)  # Initial weights
    
    # Save initial shape for later reference
    initial_size = current_z.shape[0]
    feature_size = current_z.shape[1]  # Feature dimension

    ts_list = ts.tolist()
    # -------------------------- Key Modification 1: Save historical data for all time steps (for backtracking and copying) --------------------------
    # Historical data format: [time_step0_data, time_step1_data, ...], each element is (z, lnw)
    history_z = [current_z.clone()]  # Particle feature history at initial step (t0)
    history_lnw = [current_lnw.clone()]  # Log-weight history at initial step (t0)
    idx_next = 1  # Start iteration from step 1 (t0 is already stored in history)

    while current_t <= tf + 1e-8 and idx_next < len(ts_list):
        t_tensor = torch.tensor([current_t], device=device)
        f_z, f_lnw, f_m = ode_func(t_tensor, (current_z, current_lnw, current_m))

        # 1. Euler integration to update current step particle state
        dt_step = ts_List[idx_next] - current_t
        new_z   = current_z   + f_z   * dt_step
        new_lnw = current_lnw + f_lnw * dt_step
        new_m   = current_m   + f_m   * dt_step
        current_t = ts_List[idx_next]

        w_next = torch.exp(new_lnw)
        ratio = w_next / w_prev
        current_size = new_z.shape[0]  # Current number of particles

        # -------------------------- 2. Splitting Handling: Add logic for "backtracking and copying historical data" --------------------------
        mask_split = (ratio >= 1).squeeze(-1)
        # Data after splitting: includes "original splitting particles + newly split particles"
        split_z = new_z[mask_split].clone()  # Keep original splitting particles (not discarded)
        split_lnw = new_lnw[mask_split].clone()
        split_m = new_m[mask_split].clone()
        # Record indices of original splitting particles (for subsequent backtracking and copying of history)
        original_split_indices = torch.where(mask_split)[0]
        
        if mask_split.any():
            r_split = ratio[mask_split].squeeze(-1)
            z_split_original = new_z[mask_split]  # Original splitting particles (to split based on)
            lnw_split_original = new_lnw[mask_split]
            m_split_original = new_m[mask_split]

            # Calculate the number of splits for each original splitting particle (consistent with original logic)
            floor_r = torch.floor(r_split)
            frac_r = r_split - floor_r
            rand_f = torch.rand_like(frac_r)
            mj = floor_r.int() + (rand_f < frac_r).int()  # Number of new particles split from each original particle
            valid = mj > 0

            if valid.any():
                # Filter indices of original particles that need splitting
                valid_mask = valid
                valid_mj = mj[valid_mask]  # Valid number of splits
                valid_original_indices = original_split_indices[valid_mask]  # Global indices of valid original splitting particles
                valid_z = z_split_original[valid_mask]
                valid_lnw = lnw_split_original[valid_mask]
                valid_m = m_split_original[valid_mask]

                # Generate newly split particles (current step data)
                new_split_z_list = []
                new_split_lnw_list = []
                for i in range(len(valid_mj)):
                    mj_i = valid_mj[i].item()  # Number of new particles split from the i-th original particle
                    original_idx_i = valid_original_indices[i].item()  # Global index of the i-th original particle

                    # a. Generate current-step data for newly split particles (with noise)
                    rep_z = valid_z[i].unsqueeze(0).repeat(mj_i, 1)  # Repeat mj_i times
                    rep_lnw = valid_lnw[i].unsqueeze(0).repeat(mj_i, 1)
                    noise = torch.normal(0, noise_std, size=rep_z.shape, device=device)
                    new_z_i = rep_z + noise
                    new_lnw_i = rep_lnw

                    # b. Key step: Backtrack and copy historical data of the original particle (fill historical time steps for new particles)
                    # Iterate through all historical time steps (from t0 to the step before current step) and copy the original particle's history
                    for hist_idx in range(len(history_z)):
                        # Historical data of the original particle at the historical time step
                        hist_z_original = history_z[hist_idx][original_idx_i].unsqueeze(0)
                        hist_lnw_original = history_lnw[hist_idx][original_idx_i].unsqueeze(0)
                        # Copy to the new particle's history (history of each new particle matches the original particle)
                        history_z[hist_idx] = torch.cat([history_z[hist_idx], hist_z_original.repeat(mj_i, 1)], dim=0)
                        history_lnw[hist_idx] = torch.cat([history_lnw[hist_idx], hist_lnw_original.repeat(mj_i, 1)], dim=0)

                    # c. Collect current-step data of newly split particles
                    new_split_z_list.append(new_z_i)
                    new_split_lnw_list.append(new_lnw_i)

                # Merge current-step data of all newly split particles
                if new_split_z_list:
                    new_split_z = torch.cat(new_split_z_list, dim=0)
                    new_split_lnw = torch.cat(new_split_lnw_list, dim=0)
                    new_split_m = valid_m.unsqueeze(0).repeat_interleave(valid_mj, dim=1).squeeze(0)
                    # Add newly split particles to the current-step splitting results (original splitting particles + newly split particles)
                    split_z = torch.cat([split_z, new_split_z], dim=0)
                    split_lnw = torch.cat([split_lnw, new_split_lnw], dim=0)
                    split_m = torch.cat([split_m, new_split_m], dim=0)

        # -------------------------- 3. Extinction Handling: Modify to "set to NaN instead of discarding" --------------------------
        mask_extinct = ~mask_split
        # Keep indices of all extinct particles; set dead particles to NaN (not discarded to ensure consistent time-step indices)
        extinct_z = new_z[mask_extinct].clone()
        extinct_lnw = new_lnw[mask_extinct].clone()
        extinct_m = new_m[mask_extinct].clone()
        
        if mask_extinct.any():
            r_ext = ratio[mask_extinct].squeeze(-1)
            # Generate survival mask: True = survive, False = dead (set to NaN)
            keep_mask = torch.rand_like(r_ext) < r_ext
            # Set dead particles to NaN (keep indices; no further updates in subsequent steps)
            extinct_z[~keep_mask] = torch.nan
            extinct_lnw[~keep_mask] = torch.nan
            extinct_m[~keep_mask] = torch.nan

        # -------------------------- 4. Merge: Keep all particle indices (splitting + extinction) --------------------------
        current_z = torch.cat([split_z, extinct_z], dim=0)
        current_lnw = torch.cat([split_lnw, extinct_lnw], dim=0)
        current_m = torch.cat([split_m, extinct_m], dim=0)
        w_prev = torch.exp(current_lnw)  # Update weights

        # -------------------------- 5. Save current-step data to history (for subsequent splitting backtracking) --------------------------
        history_z.append(current_z.clone())
        history_lnw.append(current_lnw.clone())
        idx_next += 1

    # -------------------------- 6. Unify shapes of all time steps (historical data is complete; directly concatenate) --------------------------
    # Historical data already contains complete time-series of all particles (original + split); directly stack into tensors
    out_z = torch.stack(history_z, dim=0)  # shape: (time_steps, num_particles, feature_size)
    out_lnw = torch.stack(history_lnw, dim=0)  # shape: (time_steps, num_particles, 1)
    
    return out_z, out_lnw


from typing import Any, Dict, List, Optional, Sequence
import json
import os, time
from ..tl.methods import ODEFunc


def _normalize_index_array(init_indices: Any, n_obs: int) -> np.ndarray:
    if init_indices is None:
        return np.asarray([], dtype=int)

    if isinstance(init_indices, str):
        tokens = [x.strip() for x in init_indices.split(",") if x.strip()]
        raw = np.asarray(tokens, dtype=object)
    else:
        raw = np.asarray(init_indices, dtype=object).reshape(-1)

    idx_list: List[int] = []
    for value in raw:
        text = str(value).strip()
        if text == "":
            continue
        try:
            num = float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid index value '{value}' in init_indices.") from exc
        if not np.isfinite(num):
            raise ValueError(f"Non-finite index value '{value}' in init_indices.")
        if abs(num - round(num)) > 1e-8:
            raise ValueError(f"Index value '{value}' is not an integer.")
        idx_list.append(int(round(num)))

    if not idx_list:
        return np.asarray([], dtype=int)

    idx = np.asarray(idx_list, dtype=int)
    if np.min(idx) < 0 or np.max(idx) >= int(n_obs):
        raise ValueError(
            f"init_indices out of range: min={int(np.min(idx))}, max={int(np.max(idx))}, n_obs={int(n_obs)}"
        )
    return idx


def _select_ode_initial_indices(
    adata,
    *,
    samples_key: str,
    n_trajectories: int,
    init_time: Optional[float],
    sampling_mode: str,
    init_cell_type: Optional[str],
    init_indices: Optional[Sequence[int]],
    cell_type_key: str,
    sampling_seed: Optional[int],
    strict_index_init_time: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if samples_key not in adata.obs:
        raise KeyError(f"{samples_key} not found in adata.obs")
    times = np.asarray(adata.obs[samples_key], dtype=float)
    unique_times = np.sort(np.unique(times))
    if unique_times.size == 0:
        raise ValueError(f"No valid values found in adata.obs['{samples_key}']")

    init_t = float(unique_times[0]) if init_time is None else float(init_time)
    time_mask = np.isclose(times, init_t, atol=1e-8, rtol=0.0)
    if not np.any(time_mask):
        raise ValueError(f"No cells found at init_time={init_t} from key '{samples_key}'")

    mode = (sampling_mode or "random").strip().lower()
    rng = np.random.default_rng(sampling_seed)

    if mode in {"random", "sample"}:
        eligible = np.where(time_mask)[0]
        replace = len(eligible) < int(n_trajectories)
        chosen = rng.choice(eligible, size=int(n_trajectories), replace=replace).astype(int)
    elif mode in {"cell_type_random", "celltype_random", "cell_type"}:
        if not init_cell_type:
            raise ValueError("sampling_mode='cell_type_random' requires init_cell_type.")
        if cell_type_key not in adata.obs:
            raise KeyError(f"{cell_type_key} not found in adata.obs")
        labels = np.asarray(adata.obs[cell_type_key]).astype(str)
        mask = time_mask & (labels == str(init_cell_type))
        eligible = np.where(mask)[0]
        if len(eligible) == 0:
            raise ValueError(
                f"No cells matched init_time={init_t} and {cell_type_key}='{init_cell_type}'."
            )
        replace = len(eligible) < int(n_trajectories)
        chosen = rng.choice(eligible, size=int(n_trajectories), replace=replace).astype(int)
    elif mode in {"index", "indices", "given_index"}:
        chosen = _normalize_index_array(init_indices, adata.n_obs)
        if chosen.size == 0:
            raise ValueError("sampling_mode='index' requires non-empty init_indices.")
        if strict_index_init_time:
            invalid = chosen[~time_mask[chosen]]
            if invalid.size > 0:
                preview = ",".join([str(int(x)) for x in invalid[:10]])
                raise ValueError(
                    "sampling_mode='index' only allows indices from init_time "
                    f"{init_t} when strict_index_init_time=True. Invalid examples: [{preview}]"
                )
    else:
        raise ValueError(
            "Unsupported sampling_mode=%s. Choose from random, cell_type_random, index." % sampling_mode
        )

    meta = {
        "sampling_mode": mode,
        "samples_key": str(samples_key),
        "init_time": float(init_t),
        "n_selected": int(chosen.shape[0]),
        "sampling_seed": (None if sampling_seed is None else int(sampling_seed)),
        "cell_type_key": str(cell_type_key),
        "init_cell_type": (None if init_cell_type is None else str(init_cell_type)),
        "strict_index_init_time": bool(strict_index_init_time),
    }
    return np.asarray(chosen, dtype=int), meta


def _extract_perturbation_targets(perturbation_cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    targets: List[str] = []
    group_name = str(perturbation_cfg.get("group_name", "")).strip()

    genes_raw = perturbation_cfg.get("genes", None)
    if genes_raw is None:
        genes_raw = perturbation_cfg.get("targets", None)

    if genes_raw is None:
        groups_raw = perturbation_cfg.get("target_groups", [])
        if isinstance(groups_raw, dict):
            groups_raw = [groups_raw]
        if isinstance(groups_raw, (list, tuple)) and len(groups_raw) > 0:
            first = groups_raw[0]
            if isinstance(first, dict):
                genes_raw = first.get("genes", [])
                if not group_name:
                    group_name = str(first.get("name", "")).strip()
            else:
                genes_raw = first

    if isinstance(genes_raw, str):
        raw_list = [genes_raw]
    elif isinstance(genes_raw, (list, tuple, np.ndarray)):
        raw_list = [str(x).strip() for x in genes_raw]
    else:
        raw_list = []

    for g in raw_list:
        gs = str(g).strip()
        if gs:
            targets.append(gs)

    if not group_name and targets:
        group_name = "+".join(targets)
    return group_name, targets


def _apply_feature_space_perturbation(
    *,
    adata,
    reference_adata=None,
    init_latent: np.ndarray,
    perturbation_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg = dict(perturbation_cfg or {})
    if not bool(cfg.get("enabled", True)):
        return np.asarray(init_latent, dtype=np.float32), {"enabled": False, "status": "disabled"}

    group_name, targets = _extract_perturbation_targets(cfg)
    if not targets:
        return np.asarray(init_latent, dtype=np.float32), {"enabled": False, "status": "no_targets"}

    z_score = cfg.get("z_score", None)
    if z_score is None:
        z_scores = cfg.get("z_scores", [])
        if isinstance(z_scores, (list, tuple, np.ndarray)) and len(z_scores) > 0:
            z_score = z_scores[0]
        else:
            z_score = 1.0
    z_score = float(z_score)

    missing_target_policy = str(cfg.get("missing_target_policy", "zero")).strip().lower()
    if missing_target_policy not in {"zero", "skip"}:
        missing_target_policy = "zero"

    adata_ref = adata if reference_adata is None else reference_adata
    if int(adata_ref.n_obs) != int(adata.n_obs):
        raise ValueError(
            "Perturbation reference adata must have the same n_obs as the simulation adata. "
            f"Got {adata_ref.n_obs} vs {adata.n_obs}."
        )
    if not np.array_equal(np.asarray(adata_ref.obs_names), np.asarray(adata.obs_names)):
        try:
            adata_ref = adata_ref[adata.obs_names].copy()
        except Exception as exc:
            raise ValueError(
                "Perturbation reference adata obs_names do not align with simulation adata."
            ) from exc

    pca_uns = adata_ref.uns.get("pca", {})
    reconstruction_raw = pca_uns.get("reconstruction_matrix", None)
    projection_raw = pca_uns.get("projection_matrix", None)
    mean_raw = pca_uns.get("input_mean", None)
    if reconstruction_raw is None:
        raise ValueError("Perturbation requires adata.uns['pca']['reconstruction_matrix'].")
    reconstruction = np.asarray(reconstruction_raw, dtype=np.float32)
    if projection_raw is None:
        projection = np.asarray(np.linalg.pinv(reconstruction), dtype=np.float32)
    else:
        projection = np.asarray(projection_raw, dtype=np.float32)
    mean_vec = np.asarray(mean_raw, dtype=np.float32) if mean_raw is not None else np.zeros(reconstruction.shape[1], dtype=np.float32)

    original_info = adata_ref.uns.get("original_gene_info", {})
    feature_names_raw = original_info.get("var_names", None)
    if feature_names_raw is None:
        feature_names = [str(x) for x in adata_ref.var_names]
    else:
        feature_names = [str(x) for x in feature_names_raw]
    feature_lut = {g: i for i, g in enumerate(feature_names)}

    x0 = np.asarray(init_latent, dtype=np.float32)
    x0_feat = np.asarray(x0 @ reconstruction + mean_vec[None, :], dtype=np.float32)
    feat_std = np.asarray(np.std(x0_feat, axis=0), dtype=np.float32)

    valid_items = []
    missing_items = []
    for gene in targets:
        if gene not in feature_lut:
            missing_items.append({"gene": gene, "status": "missing_in_feature_names"})
            continue
        idx = int(feature_lut[gene])
        std_val = float(feat_std[idx])
        if std_val < 1e-8:
            missing_items.append({"gene": gene, "status": "near_zero_std", "feature_std": std_val})
            continue
        valid_items.append({"gene": gene, "feature_index": idx, "feature_std": std_val})

    if not valid_items:
        status = "skip_no_valid_targets" if missing_target_policy == "skip" else "zero_fallback_no_valid_targets"
        return x0.copy(), {
            "enabled": True,
            "status": status,
            "group_name": group_name,
            "targets": targets,
            "z_score": z_score,
            "missing_target_policy": missing_target_policy,
            "valid_targets": [],
            "missing_targets": missing_items,
        }

    x0_pert = x0_feat.copy()
    per_target_delta = []
    for item in valid_items:
        delta = z_score * float(item["feature_std"])
        idx = int(item["feature_index"])
        x0_pert[:, idx] = x0_pert[:, idx] + delta
        per_target_delta.append(
            {
                "gene": str(item["gene"]),
                "feature_index": idx,
                "feature_std": float(item["feature_std"]),
                "delta_feature": float(delta),
            }
        )

    z0_pert = np.asarray((x0_pert - mean_vec[None, :]) @ projection, dtype=np.float32)
    meta = {
        "enabled": True,
        "status": "ok" if len(valid_items) == len(targets) else "ok_partial_targets",
        "group_name": group_name,
        "targets": targets,
        "z_score": float(z_score),
        "missing_target_policy": missing_target_policy,
        "valid_targets": valid_items,
        "missing_targets": missing_items,
        "per_target_delta": per_target_delta,
    }
    return z0_pert, meta


# ---------- Subfunction: Only responsible for integration ----------
def generate_ode_trajectories(
        model,
        adata,
        n_trajectories: int = 50,
        n_bins: int = 100,
        device: str = "cuda",
        samples_key: str = 'time_point_processed',
        exp_dir: str = None,
        split_true: str = False,
        sampling_mode: str = "random",
        init_time: Optional[float] = None,
        init_cell_type: Optional[str] = None,
        init_indices: Optional[Sequence[int]] = None,
        cell_type_key: str = "cell_type",
        sampling_seed: Optional[int] = None,
        strict_index_init_time: bool = True,
        perturbation_cfg: Optional[Dict[str, Any]] = None,
        perturbation_reference_adata = None,
        return_metadata: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    if int(n_bins) < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")
    if int(n_trajectories) < 1 and (init_indices is None or len(np.asarray(init_indices).reshape(-1)) == 0):
        raise ValueError("n_trajectories must be >=1 unless explicit init_indices are provided.")

    device = torch.device(device)
    model.to(device).eval()
    X_raw = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    chosen_init_idx, sampling_meta = _select_ode_initial_indices(
        adata,
        samples_key=samples_key,
        n_trajectories=int(n_trajectories),
        init_time=init_time,
        sampling_mode=sampling_mode,
        init_cell_type=init_cell_type,
        init_indices=init_indices,
        cell_type_key=cell_type_key,
        sampling_seed=sampling_seed,
        strict_index_init_time=bool(strict_index_init_time),
    )

    init_x_np = np.asarray(X_raw[chosen_init_idx], dtype=np.float32)
    perturbation_meta: Dict[str, Any] = {"enabled": False, "status": "not_requested"}
    if perturbation_cfg is not None:
        init_x_np, perturbation_meta = _apply_feature_space_perturbation(
            adata=adata,
            reference_adata=perturbation_reference_adata,
            init_latent=init_x_np,
            perturbation_cfg=dict(perturbation_cfg),
        )
    init_x = torch.tensor(init_x_np, dtype=torch.float32, device=device)

    from ..tl.methods import ODEFunc
    use_mass,score_use,interaction_use = check_submodules(model)

    #print(use_mass,score_use,interaction_use)
    ode_func = ODEFunc(model,use_mass=use_mass,score_use=score_use,interaction_use=interaction_use).to(device)

    all_times = np.sort(np.asarray(adata.obs[samples_key], dtype=float))
    all_times_unique = np.unique(all_times)
    init_time_used = float(sampling_meta["init_time"])
    eval_time_mask = all_times_unique >= (init_time_used - 1e-8)
    eval_times = all_times_unique[eval_time_mask]
    if eval_times.size == 0:
        eval_times = np.asarray([init_time_used], dtype=float)
    t_min, t_max = float(eval_times[0]), float(eval_times[-1])
    t_bins = torch.linspace(t_min, t_max, int(n_bins), device=device)
    t_point = torch.tensor(eval_times, dtype=torch.float32, device=device)
    dt = (t_max - t_min) / (len(t_bins) - 1)


    n_selected = int(init_x.shape[0])
    init_lnw = torch.log(torch.ones(n_selected, 1) / n_selected).to(device)
    init_m   = torch.zeros_like(init_lnw)
    initial_state = (init_x, init_lnw, init_m)
    if split_true :
        traj_x, traj_lnw  = euler_odeint_split(ode_func, initial_state, t_bins)
        point_x, point_lnw = euler_odeint_split(ode_func, initial_state, t_point)
    else :
        traj_x, traj_lnw, _ = odeint(ode_func, initial_state, t_bins,  method='euler')
        point_x, point_lnw, _ = odeint(ode_func, initial_state, t_point, method='euler', options=dict(step_size=0.2))
    traj_array  = traj_x.detach().cpu().numpy()          # (T_bins, M, D)
    traj_lnw    = traj_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)
    point_array = point_x.detach().cpu().numpy()         # (T, M, D)
    point_lnw   = point_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)
    metadata = {
        "sampling": sampling_meta,
        "init_indices": np.asarray(chosen_init_idx, dtype=int).tolist(),
        "time_points": [float(x) for x in np.asarray(eval_times, dtype=float).tolist()],
        "n_bins": int(n_bins),
        "split_true": bool(split_true),
        "perturbation": perturbation_meta,
    }
    # Saving
    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)

        print( "traj_array.shape",traj_array.shape)
        print( "ode_point.shape",point_array.shape)

        np.save(os.path.join(exp_dir, "ode_traj.npy"), traj_array)
        np.save(os.path.join(exp_dir, "ode_traj_lnw.npy"), traj_lnw)
        np.save(os.path.join(exp_dir, "ode_point.npy"),  point_array)
        np.save(os.path.join(exp_dir, "ode_point_lnw.npy"), point_lnw)
        with open(os.path.join(exp_dir, "sampling_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[generate_ode_trajectories] trajectories saved to {exp_dir}xxxxxxxxxxxx")

    if return_metadata:
        return point_array, traj_array, metadata
    return point_array, traj_array


def simulate_trajectory(adata,model, x0, sigma, time, dt, device):
    x0 = x0.requires_grad_(True)
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device, dtype=torch.float32) / x0.shape[0])
    initial_state = (x0, lnw0)

    class CytoSDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"
        def __init__(self, model, sigma=1.0):
            super().__init__()
            self.model = model
            self.sigma = sigma
        def f(self, t, y):
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            vel_in = torch.cat([z, t_expand], 1)
            drift_z = self.model.velocity_net(vel_in)
            if "score" in self.model.components:
                score, gradients = self.model.compute_score(t, z)
                drift_z += gradients
            if "growth" in self.model.components:
                drift_lnw = self.model.growth_net(vel_in)
            else:
                drift_lnw = torch.zeros_like(lnw)
            if "interaction" in self.model.components:
                interaction_force = cal_interaction(z=z,lnw=lnw,interaction_potential=self.model.interaction_net,cutoff=self.model.interaction_net.cutoff,use_mass=self.model.use_growth_in_ode_inter)
                drift_z += interaction_force

            return (drift_z, drift_lnw)
        def g(self, t, y):
            z, lnw = y
            return (torch.ones_like(z) * self.sigma,
                    torch.ones_like(lnw) * 0.0)

    sde = CytoSDE(model, sigma).to(device)

    def euler_sdeint(sde, y0, dt, ts_lst):
        y, t = y0, ts_lst[0]
        out = []
        idx = 0
        while t <= ts_lst[-1] + 1e-8:
            if t >= ts_lst[idx] - 1e-8:
                out.append(y)
                idx += 1
                if idx >= len(ts_lst):
                    break
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            f_z, f_lnw = sde.f(t_tensor, y)
            g_z, g_lnw = sde.g(t_tensor, y)
            noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
            noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
            y = (y[0] + f_z * dt + g_z * noise_z,
                 y[1] + f_lnw * dt + g_lnw * noise_lnw)
            t += dt
        while len(out) < len(ts_lst):
            out.append(out[-1])
        return torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out])

    sde_point, point_lnw = euler_sdeint(sde, initial_state, dt, time)

    sde_point = sde_point.detach().cpu().numpy()
    w_point = np.exp(point_lnw.detach().cpu().numpy())

    return sde_point, w_point

def simulate_trajectory_fir_sec (adata,model,tranmap_fir_sec, x0, sigma, time, dt, device):
    x0 = x0.requires_grad_(True)
    if  "interaction" in adata.uns['all_model']['model_config']["components"]:
        lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device, dtype=torch.float32) / adata.uns['all_model']['training_config']['defaults']["batch_size"])
    else :
        lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device, dtype=torch.float32) / x0.shape[0])
    initial_state = (x0, lnw0)

    class CytoSDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"
        def __init__(self, model, sigma=1.0):
            super().__init__()
            self.model = model
            self.sigma = sigma
        def f(self, t, y):
            z, lnw = y
            t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
            vel_in = torch.cat([z, t_expand], 1)
            drift_z = self.model.velocity_net(vel_in)
            if "score" in self.model.components:
                score, gradients = self.model.compute_score(t, z)
                drift_z += gradients
            if "growth" in self.model.components:
                drift_lnw = self.model.growth_net(vel_in)
            else:
                drift_lnw = torch.zeros_like(lnw)
            if "interaction" in self.model.components:
                interaction_force = cal_interaction(z=z,lnw=lnw,interaction_potential=self.model.interaction_net,cutoff=self.model.interaction_net.cutoff,use_mass=self.model.use_growth_in_ode_inter)
                drift_z += interaction_force

            return (drift_z, drift_lnw)
        def g(self, t, y):
            z, lnw = y
            return (torch.ones_like(z) * self.sigma,
                    torch.ones_like(lnw) * 0.0)

    sde = CytoSDE(model, sigma).to(device)

    def euler_sdeint(sde, y0, dt, ts_lst):
        y, t = y0, ts_lst[0]
        out = []
        idx = 0
        while t <= ts_lst[-1] + 1e-8:
            if t >= ts_lst[idx] - 1e-8:
                out.append(y)
                idx += 1
                if idx >= len(ts_lst):
                    break
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            f_z, f_lnw = sde.f(t_tensor, y)
            g_z, g_lnw = sde.g(t_tensor, y)
            noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
            noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
            y = (y[0] + f_z * dt + g_z * noise_z,
                 y[1] + f_lnw * dt + g_lnw * noise_lnw)
            t += dt
        while len(out) < len(ts_lst):
            out.append(out[-1])
        return torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out])

    sde_point, point_lnw = euler_sdeint(sde, initial_state, dt, time)
    
    sde_point = sde_point.detach().cpu().numpy()
    w_point = np.exp(point_lnw.detach().cpu().numpy())

    return sde_point, w_point
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Tuple

class MLPClassifier(nn.Module):
    """MLP model for cell type classification"""
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_mlp_classifier(
    adata: AnnData,
    classifyed_type: str = 'cluster',
    hidden_size: int = 128,
    train_mlp_classifier_epoches: int = 400,
    lr: float = 0.001,
    device: str = 'cuda'
) -> Tuple[MLPClassifier, LabelEncoder]:
    """
    Train MLP classifier using real data
    
    Parameters:
        adata: AnnData object containing cell data
        classifyed_type: Key of classification target in obs (default 'cluster')
        hidden_size: MLP hidden layer dimension
        train_mlp_classifier_epoches: Number of training epochs
        lr: Learning rate
        device: Training device
    
    Returns:
        Trained MLP model and label encoder
    """
    # Extract features and labels
    X = adata.X  # Assume X is the feature matrix (n_obs × n_vars)
    y = adata.obs[classifyed_type].values
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Initialize model
    input_size = X.shape[1]
    num_classes = len(label_encoder.classes_)
    model = MLPClassifier(input_size, hidden_size, num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    for epoch in range(train_mlp_classifier_epoches):
        model.train()
        X_train.requires_grad_(True)
        
        # Forward propagation
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Gradient regularization (to prevent overfitting)
        grad_outputs = torch.ones_like(outputs)
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=X_train,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_norm = grads.norm(dim=1).mean()
        reg_lambda = 1e-2
        
        # L2 regularization
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2) ** 2
        weight_decay = 1e-4
        
        # Total loss
        total_loss = loss + reg_lambda * grad_norm + weight_decay * l2_reg
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print results every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
            print(f'Epoch [{epoch+1}/{train_mlp_classifier_epoches}], Loss: {total_loss.item():.4f}, Test Acc: {accuracy:.4f}')
    
    return model, label_encoder
