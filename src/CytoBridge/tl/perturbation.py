"""
Perturbation Analysis Module for CytoBridge.

This module provides gene perturbation simulation and fate prediction functionality,
ported from DeepRUOT perturbation analysis pipeline.

Features:
- Gene expression perturbation (z-score based)
- SDE trajectory simulation using CytoBridge DynamicalModel
- Cell fate classification (KNN-based or MLP-based)
- Compatibility with datasets lacking clone information

Example usage:
    >>> from CytoBridge.tl.perturbation import (
    ...     perturb_gene_expression,
    ...     simulate_perturbation_sde,
    ...     classify_final_states
    ... )
    >>> # Perturb a gene and simulate
    >>> x_perturb = perturb_gene_expression(adata, genes=['Klf4'], z_score=2.0)
    >>> traj_unperturbed, traj_perturbed = simulate_perturbation_sde(
    ...     model, adata, x_perturb, n_sims=20, n_init=100, n_steps=40
    ... )
    >>> labels = classify_final_states(traj_perturbed[:, -1], adata, label_key='cell_type')
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def _resolve_runtime_device(device: str) -> torch.device:
    requested = str(device).strip() or "cpu"
    if requested.lower().startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


# ==============================================================================
# Gene Expression Perturbation
# ==============================================================================

def perturb_gene_expression(
    adata: AnnData,
    genes: List[str],
    z_score: Union[float, List[float], np.ndarray],
    pca_model: Optional[Any] = None,
    use_stored_pca: bool = True,
) -> np.ndarray:
    """
    Apply z-score perturbation to specified genes and project to latent space.
    
    This function perturbs gene expression by setting the specified genes to a 
    target z-score value, then projects the perturbed expression back to latent space.
    
    Args:
        adata: AnnData object with gene expression data.
        genes: List of gene names to perturb.
        z_score: Target z-score value to set for perturbed genes.
        pca_model: Optional sklearn PCA model. If None, uses stored PCA or recomputes.
        use_stored_pca: Whether to use PCA loadings stored in adata.varm['PCs'].
    
    Returns:
        Perturbed latent representation of shape (n_cells, latent_dim).
    
    Raises:
        ValueError: If no genes found or no suitable expression data available.
    
    Note:
        For datasets with only latent representation (no gene expression),
        use `perturb_latent_space` instead.
    """
    # Validate genes exist
    gene_names = list(adata.var_names)
    valid_genes = [g for g in genes if g in gene_names]
    
    if not valid_genes:
        raise ValueError(
            f"None of the specified genes found in adata.var_names. "
            f"Requested: {genes}, Available (first 10): {gene_names[:10]}"
        )
    
    if len(valid_genes) < len(genes):
        missing = set(genes) - set(valid_genes)
        logger.warning(f"Some genes not found and will be skipped: {missing}")
    
    # Get expression matrix
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    # Standardize expression (z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply perturbation
    X_perturbed = X_scaled.copy()
    gene_indices = [gene_names.index(g) for g in valid_genes]
    
    if isinstance(z_score, (list, tuple, np.ndarray)):
        if len(z_score) != len(valid_genes):
             # Try to match the original genes list length if possible, or raise error
             if len(z_score) == len(genes):
                 # Filter z_scores correspondingly
                 z_score_subset = []
                 for g, z in zip(genes, z_score):
                     if g in valid_genes:
                         z_score_subset.append(z)
                 if not z_score_subset:
                     raise ValueError("No valid genes to perturb with provided z-scores.")
                 current_z_scores = z_score_subset
             else:
                 raise ValueError(f"Length of z_score list ({len(z_score)}) does not match valid genes ({len(valid_genes)}).")
        else:
            current_z_scores = z_score
            
        for idx, z in zip(gene_indices, current_z_scores):
            X_perturbed[:, idx] = z
    else:
        # Scalar broadcast
        for idx in gene_indices:
            X_perturbed[:, idx] = z_score
    
    # Project to latent space
    if use_stored_pca and 'PCs' in adata.varm:
        # Use stored PCA loadings
        pcs = adata.varm['PCs']
        latent_dim = adata.obsm.get('X_latent', pcs).shape[1]
        if pcs.shape[1] >= latent_dim:
            pcs = pcs[:, :latent_dim]
        x_perturb = X_perturbed @ pcs
        logger.info(f"Projected perturbed expression using stored PCs to dim {latent_dim}")
    elif pca_model is not None:
        x_perturb = pca_model.transform(X_perturbed)
        logger.info("Projected perturbed expression using provided PCA model")
    else:
        # Fallback: fit new PCA
        from sklearn.decomposition import PCA
        latent_dim = adata.obsm.get('X_latent', np.zeros((1, 50))).shape[1]
        pca = PCA(n_components=latent_dim)
        pca.fit(X_scaled)
        x_perturb = pca.transform(X_perturbed)
        logger.warning("Fitted new PCA model for projection")
    
    return x_perturb.astype(np.float32)


def perturb_latent_space(
    adata: AnnData,
    dimension_indices: List[int],
    delta: float,
    latent_key: str = 'X_latent',
) -> np.ndarray:
    """
    Apply perturbation directly in latent space.
    
    This is a fallback for datasets without gene expression data.
    
    Args:
        adata: AnnData object with latent representation.
        dimension_indices: Latent dimensions to perturb.
        delta: Value to add to the specified dimensions.
        latent_key: Key in obsm for latent representation.
    
    Returns:
        Perturbed latent representation of shape (n_cells, latent_dim).
    """
    if latent_key not in adata.obsm:
        raise ValueError(f"{latent_key} not found in adata.obsm")
    
    X_latent = np.array(adata.obsm[latent_key])
    X_perturbed = X_latent.copy()
    
    for idx in dimension_indices:
        if 0 <= idx < X_perturbed.shape[1]:
            X_perturbed[:, idx] += delta
        else:
            logger.warning(f"Dimension index {idx} out of range, skipping")
    
    return X_perturbed.astype(np.float32)


# ==============================================================================
# Sampling Weights
# ==============================================================================

def compute_sampling_weights(
    adata: AnnData,
    n_cells: Optional[int] = None,
) -> np.ndarray:
    """
    Compute cell sampling weights for trajectory simulations.
    
    Uses uniform sampling for simplicity. Weights are tracked dynamically
    during SDE simulation based on growth rate.
    
    Args:
        adata: AnnData object.
        n_cells: Optional number of cells (uses adata.n_obs if None).
    
    Returns:
        Uniform sampling weights of shape (n_cells,), normalized to sum to 1.
    """
    n = n_cells if n_cells is not None else adata.n_obs
    weights = np.ones(n) / n
    return weights.astype(np.float32)


# ==============================================================================
# SDE Simulation
# ==============================================================================

class _CytoBridgeSDE(nn.Module):
    """
    SDE wrapper for CytoBridge DynamicalModel.
    
    Implements the Ito SDE: dX = f(t, X)dt + sigma * dW
    where f(t, X) = velocity(t, X) + score_gradient(t, X)
    """
    noise_type = "diagonal"
    sde_type = "ito"
    
    def __init__(self, model: nn.Module, sigma: float = 0.1):
        super().__init__()
        self.model = model
        self.sigma = sigma
    
    def f(self, t: torch.Tensor, y: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drift function."""
        z, lnw = y
        n = z.shape[0]
        
        # Expand time
        if t.dim() == 0:
            t_exp = t.expand(n, 1)
        else:
            t_exp = t.expand(n, 1)
        
        # Enable gradients for score computation (model may use autograd.grad internally)
        with torch.enable_grad():
            z_grad = z.detach().requires_grad_(True)
            
            # Compute drift from model
            outputs = self.model(t_exp, z_grad, lnw, except_interaction=True)
            
            drift = outputs.get('velocity', torch.zeros_like(z))
            
            # Add score gradient if available
            if 'score_gradient' in outputs:
                drift = drift + outputs['score_gradient']
        
        # Detach results to avoid accumulating computation graph
        drift = drift.detach()
        
        # Growth rate for log-weight update
        dlnw = outputs.get('growth', torch.zeros_like(lnw)).detach()
        
        return drift, dlnw
    
    def g(self, t: torch.Tensor, y: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffusion function."""
        z, lnw = y
        return torch.ones_like(z) * self.sigma, torch.zeros_like(lnw)


def euler_sde_integrate(
    sde: _CytoBridgeSDE,
    y0: Tuple[torch.Tensor, torch.Tensor],
    ts: torch.Tensor,
    dt: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Euler-Maruyama SDE integration.
    
    Args:
        sde: SDE object with f (drift) and g (diffusion) methods.
        y0: Initial state tuple (z0, lnw0).
        ts: Time points tensor of shape (n_steps,).
        dt: Integration step size.
    
    Returns:
        Trajectories (z_traj, lnw_traj) each of shape (n_steps, batch, dim).
    """
    z, lnw = y0
    device = z.device
    
    z_traj = [z.clone()]
    lnw_traj = [lnw.clone()]
    
    current_t = ts[0]
    
    for i in range(1, len(ts)):
        target_t = ts[i]
        
        while current_t < target_t - 1e-6:
            step = min(dt, (target_t - current_t).item())
            
            # Drift
            drift_z, drift_lnw = sde.f(current_t, (z, lnw))
            
            # Diffusion
            diff_z, diff_lnw = sde.g(current_t, (z, lnw))
            
            # Euler-Maruyama step
            noise = torch.randn_like(z)
            z = z + drift_z * step + diff_z * np.sqrt(step) * noise
            lnw = lnw + drift_lnw * step
            
            current_t = current_t + step
        
        z_traj.append(z.clone())
        lnw_traj.append(lnw.clone())
    
    return torch.stack(z_traj), torch.stack(lnw_traj)


def simulate_perturbation_sde(
    model: nn.Module,
    adata: AnnData,
    x_perturbed: np.ndarray,
    n_simulations: int = 20,
    n_init_cells: int = 100,
    n_steps: int = 40,
    sigma: float = 0.1,
    dt: float = 0.1,
    init_time: Optional[float] = None,
    end_time: Optional[float] = None,
    init_cell_type: Optional[str] = None,
    time_key: str = 'time_point_processed',
    cell_type_key: str = 'cell_type',
    sampling_weights: Optional[np.ndarray] = None,
    device: str = 'cpu',
    run_unperturbed: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run SDE simulations for perturbation analysis.
    
    Args:
        model: Trained CytoBridge DynamicalModel.
        adata: AnnData object with latent representations.
        x_perturbed: Perturbed latent representations of shape (n_cells, latent_dim).
        n_simulations: Number of simulation runs.
        n_init_cells: Number of initial cells per simulation.
        n_steps: Number of time steps per trajectory.
        sigma: SDE diffusion coefficient.
        dt: Integration step size.
        init_time: Starting time point (None = earliest from time_point_processed).
        end_time: Ending time point (None = latest from time_point_processed).
                  Times correspond to values in the 'time_key' column (default: time_point_processed).
        init_cell_type: Filter initial cells by cell type.
        time_key: Column name for time points (default: 'time_point_processed').
        cell_type_key: Column name for cell types.
        sampling_weights: Optional sampling weights per cell.
        device: Torch device ('cpu' or 'cuda').
        run_unperturbed: Whether to also simulate unperturbed trajectories.
    
    Returns:
        Dictionary with keys:
        - 'perturbed_trajectories': (n_sims, n_steps+1, n_init, latent_dim)
        - 'unperturbed_trajectories': (n_sims, n_steps+1, n_init, latent_dim) if run_unperturbed
        - 'init_indices': (n_sims, n_init) - indices of initial cells used
    """
    # Setup device
    device_obj = _resolve_runtime_device(device)
    model = model.to(device_obj)
    model.eval()
    
    # Get latent representation
    if 'X_latent' not in adata.obsm:
        raise ValueError("X_latent not found in adata.obsm")
    
    x_latent = np.array(adata.obsm['X_latent']).astype(np.float32)
    x_perturbed = x_perturbed.astype(np.float32)
    
    # Determine time range from time_key (typically time_point_processed)
    if time_key in adata.obs:
        time_vals = adata.obs[time_key].values
        unique_times = np.sort(np.unique(time_vals))
        n_times = len(unique_times)
        if init_time is None:
            init_time = float(unique_times[0])
        if end_time is None:
            end_time = float(unique_times[-1])
    else:
        logger.warning(f"{time_key} not found, using default time range 0-2")
        if init_time is None:
            init_time = 0.0
        if end_time is None:
            end_time = 2.0
        n_times = 3
    
    # Create time grid
    ts = torch.linspace(init_time, end_time, n_steps + 1, device=device_obj)
    
    # Build selection mask for initial cells
    mask = np.ones(adata.n_obs, dtype=bool)
    
    if time_key in adata.obs:
        mask &= (adata.obs[time_key].values == init_time)
    
    if init_cell_type is not None and cell_type_key in adata.obs:
        mask &= (adata.obs[cell_type_key].values == init_cell_type)
    
    eligible_indices = np.where(mask)[0]
    
    if len(eligible_indices) == 0:
        raise ValueError("No cells match the initial condition filters")
    
    if len(eligible_indices) < n_init_cells:
        logger.warning(
            f"Only {len(eligible_indices)} cells available, "
            f"requested {n_init_cells}. Using all available."
        )
        n_init_cells = len(eligible_indices)
    
    # Compute sampling weights for eligible cells
    if sampling_weights is not None:
        weights = sampling_weights[eligible_indices]
        if weights.ndim > 1:
            weights = weights.flatten()
        if weights.sum() == 0:
            logger.warning("Sum of sampling weights is zero. Using uniform weights.")
            weights = np.ones(len(weights))
        weights = weights / weights.sum()
    else:
        weights = None
    
    # Run simulations
    sde = _CytoBridgeSDE(model, sigma=sigma)
    
    all_perturbed = []
    all_perturbed_weights = []
    all_unperturbed = []
    all_init_indices = []
    
    logger.info(f"Running {n_simulations} simulations with {n_init_cells} cells each")
    
    for sim_idx in range(n_simulations):
        # Sample initial cells
        if weights is not None:
            init_idx = np.random.choice(
                eligible_indices, size=n_init_cells, 
                replace=False, p=weights
            )
        else:
            init_idx = np.random.choice(
                eligible_indices, size=n_init_cells, 
                replace=False
            )
        
        all_init_indices.append(init_idx)
        
        # Get initial states
        x0_perturbed = torch.tensor(x_perturbed[init_idx], device=device_obj)
        lnw0 = torch.log(torch.ones(n_init_cells, 1, device=device_obj) / n_init_cells)
        
        # Run perturbed simulation
        # Run perturbed simulation
        with torch.no_grad():
            traj_per, lnw_per = euler_sde_integrate(sde, (x0_perturbed, lnw0), ts, dt)
        all_perturbed.append(traj_per.cpu().numpy())
        
        # Calculate weights: exp(lnw) * n_init
        traj_weights = torch.exp(lnw_per.squeeze(-1)) * n_init_cells
        all_perturbed_weights.append(traj_weights.cpu().numpy())
        
        # Run unperturbed simulation if requested
        if run_unperturbed:
            x0_unperturbed = torch.tensor(x_latent[init_idx], device=device_obj)
            with torch.no_grad():
                traj_unp, _ = euler_sde_integrate(sde, (x0_unperturbed, lnw0), ts, dt)
            all_unperturbed.append(traj_unp.cpu().numpy())
        
        if (sim_idx + 1) % 5 == 0:
            logger.info(f"Completed simulation {sim_idx + 1}/{n_simulations}")
    
    results = {
        'perturbed_trajectories': np.array(all_perturbed),
        'perturbed_weights': np.array(all_perturbed_weights),
        'init_indices': np.array(all_init_indices),
        'time_points': ts.cpu().numpy(),
    }
    
    if run_unperturbed:
        results['unperturbed_trajectories'] = np.array(all_unperturbed)
    
    return results


# ==============================================================================
# Cell Fate Classification
# ==============================================================================

def classify_final_states(
    final_positions: np.ndarray,
    adata: AnnData,
    label_key: str = 'cell_type',
    method: str = 'knn',
    n_neighbors: int = 15,
    latent_key: str = 'X_latent',
    time_key: str = 'time_point_processed',
    final_time: Optional[float] = None,
        mlp_model: Optional[nn.Module] = None,
        label_encoder: Optional[LabelEncoder] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Classify final cell states and compute fate proportions.
    
    Args:
        final_positions: Final positions of shape (n_cells, latent_dim) or 
                         (n_sims, n_cells, latent_dim).
        adata: Reference AnnData for training classifier.
        label_key: Column name for cell type labels.
        method: Classification method ('knn' or 'mlp').
        n_neighbors: Number of neighbors for KNN.
        latent_key: Key for latent representation in adata.
        time_key: Key for time point information.
        final_time: Filter reference cells by time point (None = all or latest).
        mlp_model: Optional pre-trained MLP classifier.
        label_encoder: Optional label encoder for MLP.
    
    Returns:
        Tuple of:
        - Predicted labels array
        - Dictionary of cell type proportions
    """
    # Reshape if needed
    original_shape = final_positions.shape
    if final_positions.ndim == 3:
        n_sims, n_cells, n_dims = final_positions.shape
        final_positions_flat = final_positions.reshape(-1, n_dims)
    else:
        n_sims = 1
        n_cells = final_positions.shape[0]
        final_positions_flat = final_positions
    
    # Get reference data
    if latent_key not in adata.obsm:
        raise ValueError(f"{latent_key} not found in adata.obsm")
    
    X_ref = np.array(adata.obsm[latent_key])
    
    if label_key not in adata.obs:
        raise ValueError(f"{label_key} not found in adata.obs")
    
    y_ref = adata.obs[label_key].values
    
    # Optionally filter by time
    if final_time is not None and time_key in adata.obs:
        mask = adata.obs[time_key].values == final_time
        X_ref = X_ref[mask]
        y_ref = y_ref[mask]
    
    # Encode labels
    le = label_encoder if label_encoder is not None else LabelEncoder()
    if label_encoder is None:
        y_ref_encoded = le.fit_transform(y_ref)
    else:
        y_ref_encoded = le.transform(y_ref)
    
    # Classify
    if method == 'knn':
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_ref, y_ref_encoded)
        pred_encoded = knn.predict(final_positions_flat)
        pred_labels = le.inverse_transform(pred_encoded)
    
    elif method == 'mlp' and mlp_model is not None:
        device = next(mlp_model.parameters()).device
        with torch.no_grad():
            x_tensor = torch.tensor(final_positions_flat.astype(np.float32), device=device)
            logits = mlp_model(x_tensor)
            pred_encoded = logits.argmax(dim=1).cpu().numpy()
        pred_labels = le.inverse_transform(pred_encoded)
    
    else:
        raise ValueError(f"Invalid method: {method}. Use 'knn' or 'mlp' with a trained model.")
    
    # Reshape predictions
    if original_shape[0] != pred_labels.shape[0] and len(original_shape) == 3:
        pred_labels = pred_labels.reshape(n_sims, n_cells)
    
    # Compute proportions
    unique_labels = np.unique(le.classes_)
    proportions = {}
    
    if weights is not None:
        # Check shapes
        if weights.ndim != 1 and weights.size == pred_labels.size:
             weights = weights.flatten()
        
        if weights.shape[0] != pred_labels.shape[0]:
             # If weights shape doesn't match flattened labels, ignore or warn?
             # But here we assume correct usage. If we reshaped prediction, we reshaped correctly.
             # Actually, pred_encoded/pred_labels comes from flattened positions.
             # So weights should be flattened too.
             pass

        total_weight = weights.sum()
        for label in unique_labels:
            mask = (pred_labels.flatten() == label)
            proportions[label] = float(weights[mask].sum() / total_weight)
    else:
        for label in unique_labels:
            # Flatten to handle multi-sim case consistently
            mask = (pred_labels.flatten() == label)
            proportions[label] = float(mask.mean())
    
    return pred_labels, proportions


# ==============================================================================
# MLP Cell Type Classifier
# ==============================================================================

class MLPClassifier(nn.Module):
    """
    MLP classifier for cell type prediction in latent space.
    
    Architecture: input -> hidden -> ReLU -> hidden -> ReLU -> output
    """
    
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_cell_classifier(
    adata: AnnData,
    label_key: str,
    latent_key: str = 'X_latent',
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train an MLP classifier for cell type prediction.
    
    Args:
        adata: AnnData object with latent representation and labels.
        label_key: Column name in obs for cell type labels.
        latent_key: Key in obsm for latent representation.
        hidden_dim: Hidden layer dimension.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for Adam optimizer.
        validation_split: Fraction of data for validation.
        device: Torch device ('cpu' or 'cuda').
        save_path: Optional path to save trained model.
        random_state: Random seed for reproducibility.
    
    Returns:
        Dictionary with:
        - 'model': Trained MLPClassifier
        - 'label_encoder': Fitted LabelEncoder
        - 'classes': List of class names
        - 'train_accuracy': Final training accuracy
        - 'val_accuracy': Final validation accuracy
        - 'history': Training history (loss and accuracy per epoch)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Validate inputs
    if latent_key not in adata.obsm:
        raise ValueError(f"{latent_key} not found in adata.obsm")
    if label_key not in adata.obs:
        raise ValueError(f"{label_key} not found in adata.obs")
    
    # Get data
    X = np.array(adata.obsm[latent_key]).astype(np.float32)
    y = adata.obs[label_key].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    in_dim = X.shape[1]
    
    logger.info(f"Training MLP classifier: {in_dim} dims -> {n_classes} classes")
    logger.info(f"Classes: {list(le.classes_)}")
    
    # Train/val split
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * validation_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    X_train, y_train = X[train_idx], y_encoded[train_idx]
    X_val, y_val = X[val_idx], y_encoded[val_idx]
    
    # Convert to tensors
    device_obj = _resolve_runtime_device(device)
    X_train_t = torch.tensor(X_train, device=device_obj)
    y_train_t = torch.tensor(y_train, device=device_obj, dtype=torch.long)
    X_val_t = torch.tensor(X_val, device=device_obj)
    y_val_t = torch.tensor(y_val, device=device_obj, dtype=torch.long)
    
    # Create model
    model = MLPClassifier(in_dim, n_classes, hidden_dim).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    history = {'loss': [], 'train_acc': [], 'val_acc': []}
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle training data
        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(X_train_t))
            
            X_batch = X_train_t[start:end]
            y_batch = y_train_t[start:end]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= n_batches
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t)
            train_pred = train_logits.argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()
            
            val_logits = model(X_val_t)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()
        
        history['loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    # Final accuracy
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    logger.info(f"Training complete. Train acc: {final_train_acc:.4f}, Val acc: {final_val_acc:.4f}")
    
    # Save model if requested
    if save_path:
        checkpoint = {
            'model': model.state_dict(),
            'label_encoder': le,
            'classes': list(le.classes_),
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'n_classes': n_classes,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    return {
        'model': model,
        'label_encoder': le,
        'classes': list(le.classes_),
        'train_accuracy': final_train_acc,
        'val_accuracy': final_val_acc,
        'history': history,
    }


def load_mlp_classifier(
    load_path: str,
    device: str = 'cpu',
) -> Tuple[MLPClassifier, LabelEncoder]:
    """
    Load a trained MLP classifier from checkpoint.
    
    Args:
        load_path: Path to saved checkpoint.
        device: Torch device.
    
    Returns:
        Tuple of (model, label_encoder).
    """
    device_obj = _resolve_runtime_device(device)
    checkpoint = torch.load(load_path, map_location=device_obj, weights_only=False)
    
    model = MLPClassifier(
        checkpoint['in_dim'],
        checkpoint['n_classes'],
        checkpoint['hidden_dim']
    ).to(device_obj)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, checkpoint['label_encoder']


# ==============================================================================
# High-Level API
# ==============================================================================

def run_perturbation_analysis(
    adata: AnnData,
    model: nn.Module,
    genes: List[str],
    z_scores: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    n_simulations: int = 20,
    n_init_cells: int = 100,
    n_steps: int = 40,
    sigma: float = 0.1,
    label_key: str = 'cell_type',
    time_key: str = 'time_point_processed',
    init_time: Optional[float] = None,
    init_cell_type: Optional[str] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Run complete perturbation analysis pipeline.
    
    This is the high-level entry point that combines:
    1. Gene expression perturbation
    2. SDE trajectory simulation
    3. Cell fate classification
    4. Results aggregation
    
    Args:
        adata: AnnData with trained model and gene expression.
        model: Trained CytoBridge DynamicalModel.
        genes: List of genes to perturb.
        z_scores: List of z-score values to test.
        n_simulations: Number of simulation runs per z-score.
        n_init_cells: Number of initial cells per simulation.
        n_steps: Number of time steps.
        sigma: SDE diffusion coefficient.
        label_key: Column for cell type labels.
        time_key: Column for time points.
        init_time: Starting time point.
        init_cell_type: Filter for initial cell type.
        device: Torch device.
    
    Returns:
        Dictionary with:
        - 'genes': list of perturbed genes
        - 'z_scores': list of tested z-scores
        - 'results': dict mapping z_score -> simulation results
        - 'proportions': dict mapping z_score -> fate proportions
        - 'unperturbed_proportions': fate proportions for unperturbed case
    """
    results = {
        'genes': genes,
        'z_scores': z_scores,
        'results': {},
        'proportions': {},
        'unperturbed_proportions': None,
    }
    
    # Compute sampling weights
    weights = compute_sampling_weights(adata, time_key=time_key)
    
    # Check if gene-level perturbation is possible
    has_gene_expression = adata.n_vars > 10 and (
        any(g in adata.var_names for g in genes)
    )
    
    for z in z_scores:
        logger.info(f"Running perturbation analysis for z-score = {z}")
        
        if has_gene_expression:
            x_perturbed = perturb_gene_expression(adata, genes, z)
        else:
            # Fallback: no perturbation for z=0, warning for others
            if z != 0:
                logger.warning(
                    f"No gene expression data available. "
                    f"Cannot apply z-score perturbation."
                )
            x_perturbed = np.array(adata.obsm['X_latent']).astype(np.float32)
        
        # Run simulation
        run_unperturbed = (z == z_scores[0])  # Only run unperturbed once
        sim_results = simulate_perturbation_sde(
            model=model,
            adata=adata,
            x_perturbed=x_perturbed,
            n_simulations=n_simulations,
            n_init_cells=n_init_cells,
            n_steps=n_steps,
            sigma=sigma,
            init_time=init_time,
            init_cell_type=init_cell_type,
            time_key=time_key,
            sampling_weights=weights,
            device=device,
            run_unperturbed=run_unperturbed,
        )
        
        results['results'][z] = sim_results
        
        # Classify final states
        final_pos = sim_results['perturbed_trajectories'][:, -1, :, :]
        _, proportions = classify_final_states(
            final_pos, adata, 
            label_key=label_key, 
            method='knn'
        )
        results['proportions'][z] = proportions
        
        # Store unperturbed proportions
        if run_unperturbed and 'unperturbed_trajectories' in sim_results:
            final_pos_unp = sim_results['unperturbed_trajectories'][:, -1, :, :]
            _, unp_proportions = classify_final_states(
                final_pos_unp, adata, 
                label_key=label_key, 
                method='knn'
            )
            results['unperturbed_proportions'] = unp_proportions
    
    return results


# ==============================================================================
# Trajectory Dataset Generation
# ==============================================================================

def generate_trajectory_dataset(
    adata: AnnData,
    model: nn.Module,
    n_cells: int = 1000,
    n_steps: int = 100,
    method: str = 'sde',
    sigma: float = 0.05,
    init_time: Optional[float] = None,
    end_time: Optional[float] = None,
    init_cell_type: Optional[str] = None,
    time_key: str = 'time_point_processed',
    cell_type_key: str = 'cell_type',
    output_space: str = 'both',
    perturbed_latent: Optional[np.ndarray] = None,
    device: str = 'cpu',
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate cell trajectories and export in a format suitable for downstream analysis.
    
    This function simulates cell trajectories using the trained CytoBridge model
    and exports them in formats compatible with the cell type classifier and 
    further analysis.
    
    Args:
        adata: AnnData object with trained model.
        model: Trained CytoBridge DynamicalModel.
        n_cells: Number of initial cells to sample.
        n_steps: Number of time steps in trajectory.
        method: Simulation method ('ode' or 'sde').
        sigma: SDE diffusion coefficient (only for sde method, default 0.05).
        init_time: Starting time from time_point_processed (None = earliest).
        end_time: Ending time from time_point_processed (None = latest).
                  Times correspond to values in the time_key column.
        init_cell_type: Filter for initial cell type (optional).
        time_key: Column name for time points (default: 'time_point_processed').
        cell_type_key: Column name for cell types.
        output_space: 'latent', 'gene', or 'both'.
        perturbed_latent: Optional perturbed latent representation for perturbation analysis.
        device: Torch device.
        save_dir: Directory to save trajectory files (optional).
    
    Returns:
        Dictionary with:
        - 'trajectories_latent': (n_steps+1, n_cells, latent_dim) in latent space
        - 'trajectories_gene': (n_steps+1, n_cells, n_genes) in gene space (if output_space != 'latent')
        - 'weights': (n_steps+1, n_cells) cell weights representing effective cell counts.
          - At t=0, all weights = 1.0 (each point represents one cell)
          - At later times, weight > 1 means that position represents multiple cells (proliferation)
          - Weight < 1 means that position represents a fraction of a cell (apoptosis)
          - Total weight sum increases/decreases based on population growth/death
        - 'time_points': (n_steps+1,) time values
        - 'gene_names': List of gene names (if gene space output)
        - 'init_indices': Indices of initial cells
        - 'metadata': Dict with configuration and statistics
        - 'save_path': Path to saved file (if save_dir provided)
    """
    device_obj = _resolve_runtime_device(device)
    model = model.to(device_obj)
    model.eval()
    
    # Get latent representation
    if 'X_latent' not in adata.obsm:
        raise ValueError("X_latent not found in adata.obsm")
    
    x_latent = np.array(adata.obsm['X_latent']).astype(np.float32)
    latent_dim = x_latent.shape[1]
    
    # Determine time range from time_key (typically time_point_processed)
    if time_key in adata.obs:
        time_vals = adata.obs[time_key].values
        unique_times = np.sort(np.unique(time_vals))
        if init_time is None:
            init_time = float(unique_times[0])
        if end_time is None:
            end_time = float(unique_times[-1])
    else:
        if init_time is None:
            init_time = 0.0
        if end_time is None:
            end_time = 2.0
    
    # Create time grid
    ts = torch.linspace(init_time, end_time, n_steps + 1, device=device_obj)
    
    # Build selection mask for initial cells
    mask = np.ones(adata.n_obs, dtype=bool)
    
    if time_key in adata.obs:
        mask &= (adata.obs[time_key].values == init_time)
    
    if init_cell_type is not None and cell_type_key in adata.obs:
        mask &= (adata.obs[cell_type_key].values == init_cell_type)
    
    eligible_indices = np.where(mask)[0]
    
    if len(eligible_indices) == 0:
        raise ValueError("No cells match the initial condition filters")
    
    if len(eligible_indices) < n_cells:
        logger.warning(f"Only {len(eligible_indices)} cells available, using all")
        n_cells = len(eligible_indices)
    
    # Sample initial cells
    init_idx = np.random.choice(eligible_indices, size=n_cells, replace=False)
    
    # Get initial states
    if perturbed_latent is not None:
        x0 = torch.tensor(perturbed_latent[init_idx].astype(np.float32), device=device_obj)
    else:
        x0 = torch.tensor(x_latent[init_idx], device=device_obj)
    
    lnw0 = torch.log(torch.ones(n_cells, 1, device=device_obj) / n_cells)
    
    # Compute dt from time range and n_steps
    dt = (end_time - init_time) / n_steps
    
    logger.info(f"Generating {n_steps+1} step trajectories for {n_cells} cells using {method.upper()}, dt={dt:.4f}")
    
    # Run simulation
    if method == 'sde':
        sde = _CytoBridgeSDE(model, sigma=sigma)
        # Note: Not using no_grad because model may need gradients for score computation
        traj_latent_t, lnw_traj_t = euler_sde_integrate(sde, (x0, lnw0), ts, dt)
        traj_latent = traj_latent_t.cpu().numpy()  # (n_steps+1, n_cells, latent_dim)
        lnw_traj = lnw_traj_t.cpu().numpy()  # (n_steps+1, n_cells, 1)
    else:
        # ODE integration (simplified Euler for now)
        traj_latent = [x0.cpu().numpy()]
        lnw_list = [lnw0.cpu().numpy()]
        z = x0.clone()
        lnw = lnw0.clone()
        
        with torch.no_grad():
            for i in range(1, len(ts)):
                step = (ts[i] - ts[i-1]).item()
                t = ts[i-1]
                if t.dim() == 0:
                    t_exp = t.expand(n_cells, 1)
                else:
                    t_exp = t.expand(n_cells, 1)
                
                outputs = model(t_exp, z, lnw, except_interaction=True)
                velocity = outputs.get('velocity', torch.zeros_like(z))
                growth = outputs.get('growth', torch.zeros_like(lnw))
                z = z + velocity * step
                lnw = lnw + growth * step
                
                traj_latent.append(z.cpu().numpy())
                lnw_list.append(lnw.cpu().numpy())
        
        traj_latent = np.stack(traj_latent, axis=0)
        lnw_traj = np.stack(lnw_list, axis=0)
    
    time_points = ts.cpu().numpy()
    
    # Convert lnw to actual weights: w = exp(lnw) * n_cells
    # This scales so that at t=0, each cell has weight=1
    # Weight sum = total effective cell count (population size)
    weights_raw = np.exp(lnw_traj.squeeze(-1))  # (n_steps+1, n_cells)
    weights = weights_raw * n_cells  # Scale so initial weights are ~1.0 per cell
    
    results = {
        'trajectories_latent': traj_latent,
        'weights': weights,  # (n_steps+1, n_cells) - effective cell counts
        'time_points': time_points,
        'init_indices': init_idx,
        'latent_dim': latent_dim,
        'metadata': {
            'n_cells': n_cells,
            'n_steps': n_steps,
            'method': method,
            'sigma': sigma if method == 'sde' else None,
            'init_time': init_time,
            'end_time': end_time,
            'init_cell_type': init_cell_type,
            'output_space': output_space,
            'shape_latent': traj_latent.shape,
            'shape_weights': weights.shape,
            'weights_description': 'Cell weights representing effective cell counts. Weight=1 means one cell, weight=2 means position represents 2 cells. Sum of weights = total population size.',
        }
    }
    
    # Project to gene space if requested
    if output_space in ['gene', 'both']:
        if 'PCs' in adata.varm:
            pcs = adata.varm['PCs']
            if pcs.shape[1] >= latent_dim:
                pcs = pcs[:, :latent_dim]
            
            # Flatten, project, reshape
            n_t, n_c, n_d = traj_latent.shape
            traj_flat = traj_latent.reshape(-1, n_d)
            traj_gene_flat = traj_flat @ pcs.T
            traj_gene = traj_gene_flat.reshape(n_t, n_c, -1)
            
            results['trajectories_gene'] = traj_gene
            results['gene_names'] = list(adata.var_names)
            results['metadata']['shape_gene'] = traj_gene.shape
            results['metadata']['n_genes'] = traj_gene.shape[2]
            
            logger.info(f"Projected to gene space: {traj_gene.shape}")
        else:
            logger.warning("PCA loadings not found, skipping gene-space projection")
            if output_space == 'gene':
                results['trajectories_gene'] = None
                results['gene_names'] = None
    
    # Save if directory provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as npz for easy loading
        save_path = os.path.join(save_dir, "trajectory_dataset.npz")
        
        save_dict = {
            'trajectories_latent': results['trajectories_latent'],
            'weights': results['weights'],  # (n_steps+1, n_cells)
            'time_points': results['time_points'],
            'init_indices': results['init_indices'],
        }
        
        if 'trajectories_gene' in results and results['trajectories_gene'] is not None:
            save_dict['trajectories_gene'] = results['trajectories_gene']
        
        np.savez(save_path, **save_dict)
        
        # Save gene names as text file
        if 'gene_names' in results and results['gene_names'] is not None:
            gene_names_path = os.path.join(save_dir, "gene_names.txt")
            with open(gene_names_path, 'w') as f:
                for g in results['gene_names']:
                    f.write(f"{g}\n")
            results['gene_names_path'] = gene_names_path
        
        # Save metadata as JSON
        import json
        metadata_path = os.path.join(save_dir, "trajectory_metadata.json")
        
        # Convert numpy types to python types
        metadata_save = {}
        for k, v in results['metadata'].items():
            if isinstance(v, np.ndarray):
                metadata_save[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                metadata_save[k] = float(v)
            elif isinstance(v, tuple):
                metadata_save[k] = list(v)
            else:
                metadata_save[k] = v
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_save, f, indent=2)
        
        results['save_path'] = save_path
        results['metadata_path'] = metadata_path
        
        logger.info(f"Saved trajectory dataset to {save_path}")
    
    return results


def load_trajectory_dataset(load_dir: str) -> Dict[str, Any]:
    """
    Load a saved trajectory dataset.
    
    Args:
        load_dir: Directory containing saved trajectory files.
    
    Returns:
        Dictionary with trajectories, time points, gene names, and metadata.
    """
    import os
    import json
    
    npz_path = os.path.join(load_dir, "trajectory_dataset.npz")
    metadata_path = os.path.join(load_dir, "trajectory_metadata.json")
    gene_names_path = os.path.join(load_dir, "gene_names.txt")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Trajectory dataset not found at {npz_path}")
    
    data = np.load(npz_path)
    
    results = {
        'trajectories_latent': data['trajectories_latent'],
        'time_points': data['time_points'],
        'init_indices': data['init_indices'],
    }
    
    if 'weights' in data:
        results['weights'] = data['weights']
    
    if 'trajectories_gene' in data:
        results['trajectories_gene'] = data['trajectories_gene']
    
    if os.path.exists(gene_names_path):
        with open(gene_names_path, 'r') as f:
            results['gene_names'] = [line.strip() for line in f]
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            results['metadata'] = json.load(f)
    
    return results
