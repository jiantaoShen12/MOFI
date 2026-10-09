import torch
import torch.nn as nn
from torchdiffeq import odeint

__all__ = ["neural_ode_step", "MongeMapSimu", "ODEFuncSimu"]


def neural_ode_step(ode_func, x0, lnw0, t0, t1, device):
    time = torch.tensor([t0, t1], dtype=torch.float32, device=device)
    e0 = torch.zeros_like(lnw0, device=device)
    initial_state = (x0, lnw0, e0)
    x_t, lnw_t, e_t = odeint(
        ode_func,
        initial_state,
        time,
        method="euler",
        options={"step_size": 0.05},
    )
    return x_t[-1], lnw_t[-1], e_t[-1]


class MongeMapSimu:
    def __init__(
        self,
        sigma_hill_A: float = 0.05,
        sigma_hill_B: float = 0.15,
        x_center: float = 2.3,
        y_center: float = 0.6,
    ):
        self.sigma_hill_A = float(sigma_hill_A)
        self.sigma_hill_B = float(sigma_hill_B)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.dtype = torch.float64
        self.output_dim = 3

    def map(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.ndim != 2:
            raise ValueError(f"xy must be 2D, got ndim={xy.ndim}")
        if xy.shape[1] < 2:
            raise ValueError(f"xy must have at least 2 dims, got shape={tuple(xy.shape)}")

        x = xy[:, 0].to(dtype=self.dtype)
        y = xy[:, 1].to(dtype=self.dtype)
        s = (
            (x - self.x_center) ** 2 / (2.0 * self.sigma_hill_A)
            + (y - self.y_center) ** 2 / (2.0 * self.sigma_hill_B)
        )
        t3 = torch.exp(-s)

        return torch.cat(
            [
                x.unsqueeze(1),
                y.unsqueeze(1),
                t3.unsqueeze(1),
            ],
            dim=1,
        )

    def gradient(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.ndim != 2:
            raise ValueError(f"xy must be 2D, got ndim={xy.ndim}")
        batchsize, dim = xy.shape
        if dim < 2:
            raise ValueError(f"xy must have at least 2 dims, got shape={tuple(xy.shape)}")

        x = xy[:, 0].to(dtype=self.dtype)
        y = xy[:, 1].to(dtype=self.dtype)
        s = (
            (x - self.x_center) ** 2 / (2.0 * self.sigma_hill_A)
            + (y - self.y_center) ** 2 / (2.0 * self.sigma_hill_B)
        )
        t3 = torch.exp(-s)

        grad_t = torch.zeros(
            (self.output_dim, dim, batchsize),
            dtype=self.dtype,
            device=xy.device,
        )
        grad_t[0, 0, :] = 1.0
        grad_t[1, 1, :] = 1.0
        grad_t[2, 0, :] = -t3 * (x - self.x_center) / self.sigma_hill_A
        grad_t[2, 1, :] = -t3 * (y - self.y_center) / self.sigma_hill_B
        return grad_t

    def compute_vTv(self, u: torch.Tensor, grad_t: torch.Tensor) -> torch.Tensor:
        if u.ndim != 2:
            raise ValueError(f"u must be 2D, got ndim={u.ndim}")
        if grad_t.ndim != 3:
            raise ValueError(f"grad_t must be 3D, got ndim={grad_t.ndim}")

        batchsize_u, dim_u = u.shape
        output_dim_g, dim_g, batchsize_g = grad_t.shape
        if batchsize_u != batchsize_g:
            raise ValueError(f"u batchsize {batchsize_u} != grad_t batchsize {batchsize_g}")
        if dim_u != dim_g:
            raise ValueError(f"u dim {dim_u} != grad_t dim {dim_g}")
        if output_dim_g != self.output_dim:
            raise ValueError(f"grad_t output dim {output_dim_g} != expected {self.output_dim}")

        u = u.to(dtype=self.dtype, device=grad_t.device)
        u_reshaped = u.T
        v = torch.einsum("iod,od->id", grad_t, u_reshaped)
        result = torch.einsum("id,id->d", v, v).unsqueeze(1)
        return result


class ODEFuncSimu(nn.Module):
    def __init__(
        self,
        model,
        sigma: float = 0.05,
        model_stra=None,
        use_mass: bool = False,
        score_use: bool = False,
        interaction_use: bool = False,
        use_monge_energy_regularizer: bool = True,
        monge_map_kwargs=None,
        multi_alpha: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.sigma = float(sigma)
        self.multi_alpha = float(multi_alpha)
        self.use_monge_energy_regularizer = bool(use_monge_energy_regularizer)
        self.monge_map_kwargs = dict(monge_map_kwargs or {})
        self.monge = MongeMapSimu(**self.monge_map_kwargs)

        if model_stra is not None:
            self.use_mass = "g" in model_stra
            self.score_use = "s" in model_stra
            self.interaction_use = "i" in model_stra
        else:
            self.use_mass = bool(use_mass)
            self.score_use = bool(score_use)
            self.interaction_use = bool(interaction_use)

    def forward(self, t, state):
        x, lnw, m = state
        batch_size = x.shape[0]

        outputs = self.model(t, x, lnw)
        v = outputs["velocity"]

        if self.use_mass and "growth" in outputs:
            g = outputs["growth"]
        else:
            g = torch.zeros(batch_size, 1, device=x.device)

        if (self.score_use and "score" in outputs) and (
            not self.interaction_use
            or "interaction" not in outputs
            or self.model.interaction_net.cutoff == 0
        ):
            s = outputs["score"]
            grad_s = outputs["score_gradient"]
            v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
            grad_s_norm_sq = torch.norm(grad_s, p=2, dim=1, keepdim=True) ** 2
            de_dt = (
                v_norm_sq / 2.0
                + grad_s_norm_sq / 2.0
                - (0.5 * self.sigma**2 * g + s * g)
                + g**2
            ) * torch.exp(lnw)
        elif (self.score_use and "score" in outputs) and (self.interaction_use and "interaction" in outputs):
            grad_s = outputs["score_gradient"]
            norm_grad_s = torch.norm(grad_s, p=2, dim=1, keepdim=True).requires_grad_(True)
            vel_norm = torch.norm(v, p=2, dim=1, keepdim=True)
            de_dt = (
                vel_norm**2 / 2.0
                + norm_grad_s**2 / 2.0
                + vel_norm * torch.norm(grad_s, p=2, dim=1, keepdim=True)
                + g**2
            ) * torch.exp(lnw)
        else:
            v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
            reg_term = 0.0
            if self.use_monge_energy_regularizer and self.multi_alpha > 0:
                grad_t = self.monge.gradient(x)
                reg_term = self.monge.compute_vTv(v, grad_t) * self.multi_alpha
            de_dt = (0.5 * (v_norm_sq + reg_term) + g**2) * torch.exp(lnw)
            de_dt = m + de_dt

        if self.interaction_use and "interaction" in outputs:
            v = v + outputs["interaction"]

        dx_dt = v
        dlnw_dt = g
        dm_dt = de_dt

        self._check_dim_consistency(x, dx_dt, "x")
        self._check_dim_consistency(lnw, dlnw_dt, "lnw")
        self._check_dim_consistency(m, dm_dt, "m")

        return dx_dt.float(), dlnw_dt.float(), dm_dt.float()

    @staticmethod
    def _check_dim_consistency(var, deriv, name: str):
        if var.shape != deriv.shape:
            raise AssertionError(
                f"Dimension mismatch for {name}: var shape {tuple(var.shape)} != deriv shape {tuple(deriv.shape)}"
            )
