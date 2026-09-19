# • 新增/修改如下：

#   - 新增 moscot 映射实现
#     moscot_map.py
#   - 新增映射工厂（ae/moscot 统一创建）
#     transport_factory.py
#   - 导出新模块
#     init.py
#   - 训练器改为走工厂，并加 update_transport 兼容分支
#     trainer.py
#   - fit_again 保存 tranmap checkpoint 时加保护（moscot 无可训练 model 时跳过）
#     fit.py
#   - run_pipeline.py 增加 map.mapper_type / map.mapper_kwargs 配置支持，并在方向选择、跨空间评估、可视化都接入工厂
#     run_pipeline.py
#   - 新增“从两个 h5ad 直接求 moscot coupling 并导出 npz”的脚本
#     build_moscot_coupling.py

#   执行产物（本次实际跑出来）：

#   - coupling：/tmp/axolotl_moscot_coupling_3000.npz

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch


class MoscotTransportMap:
    """
    Optional transport-map adapter backed by a precomputed moscot coupling.

    Expected file format (`.npz`):
    - `source_latent`: (n_source, in_dim)
    - `target_latent`: (n_target, out_dim)
    - `coupling`: (n_source, n_target)
    """

    supports_training = False

    def __init__(
        self,
        model_path: Optional[str],
        input_dim1: int,
        input_dim2: int,
        mode: Tuple[int, int],
        device: str = "cpu",
        hidden_dim: int = 16,
        kernel_scale: Optional[float] = None,
        chunk_size: int = 2048,
        source_key: str = "source_latent",
        target_key: str = "target_latent",
        coupling_key: str = "coupling",
        reverse_coupling_key: Optional[str] = None,
    ):
        del hidden_dim  # kept for API compatibility with TransportMap
        self.device = torch.device(device)
        self.input_dim1 = int(input_dim1)
        self.input_dim2 = int(input_dim2)
        self.mode = self._parse_mode(mode)
        self.in_dim = self.input_dim1 if self.mode[0] == 1 else self.input_dim2
        self.out_dim = self.input_dim1 if self.mode[1] == 1 else self.input_dim2
        self.model = None  # keep attribute for compatibility checks
        self.chunk_size = int(chunk_size)

        if model_path is None:
            raise ValueError("moscot mapper requires `Map_model_path` pointing to a coupling file (.npz).")
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"moscot coupling file not found: {model_path}")

        payload = np.load(model_path, allow_pickle=True)
        raw_source_latent = np.asarray(payload[source_key], dtype=np.float32)
        raw_target_latent = np.asarray(payload[target_key], dtype=np.float32)
        raw_coupling = np.asarray(payload[coupling_key], dtype=np.float64)

        if raw_source_latent.ndim != 2 or raw_target_latent.ndim != 2 or raw_coupling.ndim != 2:
            raise ValueError("Invalid moscot payload shapes, expected 2D arrays for source/target/coupling.")

        if raw_coupling.shape != (raw_source_latent.shape[0], raw_target_latent.shape[0]):
            if raw_coupling.shape == (raw_target_latent.shape[0], raw_source_latent.shape[0]):
                raw_coupling = raw_coupling.T
            else:
                raise ValueError(
                    "coupling shape mismatch: "
                    f"expected {(raw_source_latent.shape[0], raw_target_latent.shape[0])}, got {raw_coupling.shape}."
                )

        raw_coupling_rev = None
        reverse_coupling = None
        if reverse_coupling_key is not None and reverse_coupling_key in payload.files:
            reverse_coupling = np.asarray(payload[reverse_coupling_key], dtype=np.float64)
        if reverse_coupling is None:
            raw_coupling_rev = raw_coupling.T
        else:
            if reverse_coupling.shape == (raw_target_latent.shape[0], raw_source_latent.shape[0]):
                raw_coupling_rev = reverse_coupling
            elif reverse_coupling.shape == (raw_source_latent.shape[0], raw_target_latent.shape[0]):
                raw_coupling_rev = reverse_coupling.T
            else:
                raise ValueError(
                    "reverse coupling shape mismatch: "
                    f"expected {(raw_target_latent.shape[0], raw_source_latent.shape[0])} or "
                    f"{(raw_source_latent.shape[0], raw_target_latent.shape[0])}, got {reverse_coupling.shape}."
                )

        raw_bary_source_to_target = self._barycentric_projection(raw_coupling, raw_target_latent)
        raw_bary_target_to_source = self._barycentric_projection(raw_coupling_rev, raw_source_latent)

        if self.mode == (1, 2):
            source_latent = raw_source_latent
            target_latent = raw_target_latent
            forward_bary = raw_bary_source_to_target
            reverse_source_latent = raw_target_latent
            reverse_bary = raw_bary_target_to_source
        elif self.mode == (2, 1):
            source_latent = raw_target_latent
            target_latent = raw_source_latent
            forward_bary = raw_bary_target_to_source
            reverse_source_latent = raw_source_latent
            reverse_bary = raw_bary_source_to_target
        else:
            raise ValueError(f"Unsupported mode for moscot mapper: {self.mode}. Use (1,2) or (2,1).")

        if source_latent.shape[1] != self.in_dim:
            raise ValueError(
                f"source_latent dim mismatch: expected {self.in_dim}, got {source_latent.shape[1]}."
            )
        if target_latent.shape[1] != self.out_dim:
            raise ValueError(
                f"target_latent dim mismatch: expected {self.out_dim}, got {target_latent.shape[1]}."
            )

        self.source_latent = torch.as_tensor(source_latent, dtype=torch.float32, device=self.device)
        self.target_latent = torch.as_tensor(target_latent, dtype=torch.float32, device=self.device)
        self.forward_barycenter = torch.as_tensor(forward_bary, dtype=torch.float32, device=self.device)
        self.reverse_source_latent = torch.as_tensor(reverse_source_latent, dtype=torch.float32, device=self.device)
        self.reverse_barycenter = torch.as_tensor(reverse_bary, dtype=torch.float32, device=self.device)

        if kernel_scale is None:
            self.kernel_scale = self._estimate_kernel_scale(self.source_latent)
        else:
            self.kernel_scale = max(float(kernel_scale), 1e-8)

        print("✓ moscot transport map initialized")
        print(f"  mode: {self.mode[0]}->{self.mode[1]}")
        print(f"  source cells: {self.source_latent.shape[0]}, target cells: {self.target_latent.shape[0]}")
        print(f"  kernel_scale: {self.kernel_scale:.6f}")

    @staticmethod
    def _parse_mode(mode: Any) -> Tuple[int, int]:
        if isinstance(mode, str):
            import ast

            mode = ast.literal_eval(mode)
        if not isinstance(mode, tuple) or len(mode) != 2:
            raise ValueError(f"Invalid mode: {mode}. Expected tuple like (1, 2).")
        return int(mode[0]), int(mode[1])

    @staticmethod
    def _barycentric_projection(coupling: np.ndarray, target_latent: np.ndarray) -> np.ndarray:
        row_mass = coupling.sum(axis=1, keepdims=True)
        row_mass[row_mass <= 0] = 1.0
        return (coupling @ target_latent) / row_mass

    @staticmethod
    def _estimate_kernel_scale(anchor: torch.Tensor) -> float:
        if anchor.shape[0] < 2:
            return 1.0
        n = int(min(anchor.shape[0], 512))
        sample = anchor[:n]
        with torch.no_grad():
            dist2 = torch.cdist(sample, sample).pow(2)
            mask = dist2 > 0
            if not torch.any(mask):
                return 1.0
            return float(torch.median(dist2[mask]).item() + 1e-8)

    def _to_tensor(self, x: Any, requires_grad: bool = False) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device=self.device, dtype=torch.float32)
        if requires_grad:
            if not x.requires_grad:
                x = x.clone().detach().requires_grad_(True)
            else:
                x.requires_grad_(True)
        return x

    def _kernel_project(
        self, x: torch.Tensor, anchor: torch.Tensor, value: torch.Tensor, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        chunk_size = self.chunk_size if chunk_size is None else int(chunk_size)
        outputs = []
        for start in range(0, x.shape[0], chunk_size):
            x_chunk = x[start : start + chunk_size]
            dist2 = torch.cdist(x_chunk, anchor).pow(2)
            weights = torch.softmax(-dist2 / self.kernel_scale, dim=1)
            outputs.append(weights @ value)
        return torch.cat(outputs, dim=0)

    def T(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x, requires_grad=isinstance(x, torch.Tensor) and x.requires_grad)
        return self._kernel_project(x_t, self.source_latent, self.forward_barycenter)

    def T_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.T(x)

    def T_rev(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x, requires_grad=isinstance(x, torch.Tensor) and x.requires_grad)
        return self._kernel_project(x_t, self.reverse_source_latent, self.reverse_barycenter)

    def grad_T(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x, requires_grad=True)
        output = self._kernel_project(x_t, self.source_latent, self.forward_barycenter, chunk_size=x_t.shape[0])
        batch_size = x_t.shape[0]
        jacobian = torch.zeros(batch_size, self.out_dim, self.in_dim, device=self.device)
        for i in range(self.out_dim):
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1.0
            grad_input = torch.autograd.grad(
                outputs=output,
                inputs=x_t,
                grad_outputs=grad_output,
                retain_graph=(i < self.out_dim - 1),
                create_graph=False,
            )[0]
            jacobian[:, i, :] = grad_input
        return jacobian

    def sec_vecolity(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            v = torch.FloatTensor(v)
        jac = self.grad_T(x)
        jacv = torch.bmm(jac, v.to(self.device).unsqueeze(2))
        return jacv.squeeze(2)

    def energy_Secondary(self, x: torch.Tensor, v: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            v = torch.FloatTensor(v)
        jac = self.grad_T(x)
        jacv = torch.bmm(jac, v.to(self.device).unsqueeze(2)).squeeze(2)
        energy = torch.sum(jacv ** 2, dim=1, keepdim=True)
        return float(alpha) * energy
