import torch
from torchdiffeq import odeint
import torch.nn as nn
from typing import Tuple

__all__ = ['ODEFunc', 'MongeMap_simu']


def neural_ode_step(ODE_func, x0, lnw0, t0, t1, device):
    time = torch.Tensor([t0, t1]).to(device)
    e0 = torch.zeros_like(lnw0).to(device)
    initial_state = (x0, lnw0, e0)
    x_t, lnw_t, e_t = odeint(ODE_func, initial_state, time, method='euler', options=dict(step_size=0.05))
    return x_t[-1], lnw_t[-1], e_t[-1]


class MongeMap_simu:
    def __init__(
        self,
        sigma_hill_A: float = 0.05,
        sigma_hill_B: float = 0.05,
        x_center: float = 2.0,
        y_center: float = 1.5,
    ):
        """
        初始化Monge映射（支持任意输入维度dim，优化极大值处理）
        参数:
            sigma: 映射尺度参数（默认0.15，控制指数项陡峭程度）
            max_exp: 指数项上限（exp(max_exp)≈1e304，不超float64上限，避免极端值）
        """
        self.sigma_hill_A = sigma_hill_A
        self.sigma_hill_B = sigma_hill_B
        self.dtype = torch.float64
        self.output_dim = 3  # 输出维度固定为3
        self.x_center = x_center
        self.y_center = y_center

    def map(self, xy: torch.Tensor) -> torch.Tensor:
        """
        计算Monge映射 T: R^dim → R^3
        输入:
            xy: (batchsize, dim) 张量（每行1个样本，前两列必须是x/y）
        输出:
            T: (batchsize, 3) 张量
        """
        if xy.ndim != 2:
            raise ValueError(f"xy必须是2维张量，当前维度: {xy.ndim}")
        batchsize, dim = xy.shape
        if dim < 2:
            raise ValueError(f"输入维度dim必须≥2，当前dim: {dim}")

        x = xy[:, 0].to(dtype=self.dtype)
        y = xy[:, 1].to(dtype=self.dtype)

        s = (x - self.x_center ) ** 2 / (2 * self.sigma_hill_A) + (y - self.y_center) ** 2 / (2 * self.sigma_hill_B)
        t3 = torch.exp(-s)

        T = torch.cat([
            x.unsqueeze(1),
            y.unsqueeze(1),
            t3.unsqueeze(1)
        ], dim=1)
        return T

    def gradient(self, xy: torch.Tensor) -> torch.Tensor:
        """
        计算∇T（雅可比矩阵）
        输入:
            xy: (batchsize, dim) 张量
        输出:
            grad_T: (output_dim, dim, batchsize) 张量
                    grad_T[:, :, k] → 第k个样本的3×dim雅可比矩阵
        """
        if xy.ndim != 2:
            raise ValueError(f"xy必须是2维张量，当前维度: {xy.ndim}")
        batchsize, dim = xy.shape
        if dim < 2:
            raise ValueError(f"输入维度dim必须≥2，当前dim: {dim}")

        x = xy[:, 0].to(dtype=self.dtype)
        y = xy[:, 1].to(dtype=self.dtype)
        s = (x - self.x_center ) ** 2 / (2 * self.sigma_hill_A) + (y - self.y_center) ** 2 / (2 * self.sigma_hill_B)
        t3 = torch.exp(-s)

        # 初始化雅可比矩阵（保持与输入相同设备）
        grad_T = torch.zeros(
            (self.output_dim, dim, batchsize),
            dtype=self.dtype,
            device=xy.device
        )

        # T1对各维度偏导（仅第0维为1）
        grad_T[0, 0, :] = 1.0
        # T2对各维度偏导（仅第1维为1）
        grad_T[1, 1, :] = 1.0
        # T3对各维度偏导（仅前两维非零）
        grad_T[2, 0, :] = - t3 * (x - self.x_center) / self.sigma_hill_A
        grad_T[2, 1, :] = - t3 * (y - self.y_center) / self.sigma_hill_B

        return grad_T

    def compute_vTv(self, u: torch.Tensor, grad_T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心计算：(∇T · u)⊤ · (∇T · u)
        输入:
            u: (batchsize, dim) 张量
            grad_T: (3, dim, batchsize) 张量
        输出:
            result_core: (batchsize,) 原始结果
            result_display: (batchsize,) 优化显示结果（极端值替换为nan）
        """
        # 维度校验
        if u.ndim != 2:
            raise ValueError(f"u必须是2维张量，当前维度: {u.ndim}")
        batchsize_u, dim_u = u.shape
        output_dim_g, dim_g, batchsize_g = grad_T.shape

        if batchsize_u != batchsize_g:
            raise ValueError(f"u批量数({batchsize_u})与grad_T批量数({batchsize_g})不匹配")
        if dim_u != dim_g:
            raise ValueError(f"u维度({dim_u})与grad_T输入维度({dim_g})不匹配")
        if output_dim_g != self.output_dim:
            raise ValueError(f"grad_T输出维度({output_dim_g})必须为{self.output_dim}")

        # 统一数据类型和设备
        u = u.to(dtype=self.dtype, device=grad_T.device)

        # 计算∇T · u（3×dim × dim×1 = 3×1，按批量并行）
        u_reshaped = u.T  # (dim, batchsize)
        v = torch.einsum("iod, od -> id", grad_T, u_reshaped)  # v: (3, batchsize)

        # 计算点积：(3×1)⊤ · (3×1) = 标量
        result_core = torch.einsum("id, id -> d", v, v)  # (batchsize,)
        result_core = result_core.unsqueeze(1)


        return result_core

class ODEFunc(nn.Module):
    def __init__(self, model, tranmap=None,tranmap_simu=None,multi_alpha=0,sigma=0.05, model_stra=None,
                 use_mass=False, score_use=False, interaction_use=False):
        super(ODEFunc, self).__init__()
        self.model = model
        self.tranmap = tranmap
        self.tranmap_simu = tranmap_simu
        self.sigma = sigma
        self.multi_alpha = multi_alpha

        if model_stra is not None:
            self.use_mass = 'g' in model_stra
            self.score_use = 's' in model_stra
            self.interaction_use = 'i' in model_stra
        else:
            # 否则使用单独传入的参数
            self.use_mass = use_mass
            self.score_use = score_use
            self.interaction_use = interaction_use

    def forward(self, t, state):

        x, lnw, m = state
        batch_size = x.shape[0]

        outputs = self.model(t, x, lnw)
        v = outputs['velocity']

        if self.use_mass and 'growth' in outputs:
            g = outputs['growth']
        else:
            g = torch.zeros(batch_size, 1, device=x.device)

        if (self.score_use and 'score' in outputs) and (
                not self.interaction_use or
                not 'interaction' in outputs or
                self.model.interaction_net.cutoff == 0
        ):
            s = outputs['score']
            grad_s = outputs['score_gradient']

            v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
            grad_s_norm_sq = torch.norm(grad_s, p=2, dim=1, keepdim=True) ** 2

            de_dt = (v_norm_sq / 2
                     + grad_s_norm_sq / 2
                     - (0.5 * self.sigma ** 2 * g + s * g)
                     + g ** 2) * torch.exp(lnw)
        # print("V+G+s/V+s")

        elif (self.score_use and 'score' in outputs) and (self.interaction_use and 'interaction' in outputs):
            s = outputs['score']
            grad_s = outputs['score_gradient']
            norm_grad_s = torch.norm(grad_s, p=2, dim=1).unsqueeze(1).requires_grad_(True)

            de_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) +
                     (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2,
                                                                                                  dim=1).unsqueeze(
                        1) + g ** 2) * torch.exp(lnw)
            # print("V+S+G+I/V+S+I")
        else:
            if self.tranmap != None:
                energy_second= self.tranmap.energy_Secondary(x, v,alpha = self.multi_alpha)  # 仅取第一个返回值（张量）
                v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2

                de_dt = (0.5 * (v_norm_sq + energy_second) + g ** 2) * torch.exp(lnw)
                de_dt = m+de_dt
            elif self.tranmap_simu != None:
                T = self.tranmap_simu.map(x)
                grad_T = self.tranmap_simu.gradient(x)
                energy_second= self.tranmap_simu.compute_vTv(v, grad_T)  # 仅取第一个返回值（张量）

                v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
                energy_second = energy_second * self.multi_alpha

                de_dt = (0.5 * (v_norm_sq + energy_second) + g ** 2) * torch.exp(lnw)
                de_dt = m+de_dt
            else:
                v_norm_sq = torch.norm(v, p=2, dim=1, keepdim=True) ** 2
                de_dt = (0.5 * v_norm_sq + g ** 2) * torch.exp(lnw)
            # print("V/V+G/V+I/V+G+I")

        if self.interaction_use and 'interaction' in outputs:
            net_force = outputs['interaction']
            v = v + net_force
            # print("net_force = outputs['interaction']v = v + net_force")

        dx_dt = v
        dlnw_dt = g
        dm_dt = de_dt

        self._check_dim_consistency(x, dx_dt, "x")
        self._check_dim_consistency(lnw, dlnw_dt, "lnw")
        self._check_dim_consistency(m, dm_dt, "m")

        return dx_dt.float(), dlnw_dt.float(), dm_dt.float()

    def _check_dim_consistency(self, var, deriv, name):
        """Check dimension consistency"""
        assert var.shape == deriv.shape, \
            f"Dimension mismatch: {name} shape {var.shape} and derivative shape {deriv.shape} are not consistent"
