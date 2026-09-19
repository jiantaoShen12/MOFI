"""
传输映射 (Transport Maps)
实现 1to1, 1to2, 2to1, 2to2 四种映射，支持雅可比矩阵计算和次空间能量计算
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path
from CytoBridge.Map.tl.models import SplicedAutoEncoder
import numpy as np


def _resolve_runtime_device(device: str) -> torch.device:
    requested = str(device).strip() or "cpu"
    if requested.lower().startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


class TransportMap:
    """
    传输映射基类
    支持函数求值、雅可比矩阵计算和次空间能量计算
    """
    
    def __init__(
        self,
        model_path: str,
        input_dim1: int,
        input_dim2: int,
        mode: Tuple[int, int],
        hidden_dim: int = 16,
        device: str = 'cpu'
    ):
        """
        初始化传输映射
        
        参数:
            model_path: 训练好的模型路径
            input_dim1: RNA 潜在维度
            input_dim2: Protein 潜在维度
            mode: 转换模式 (in_domain, out_domain)
                  (1,1): RNA→RNA
                  (1,2): RNA→Protein
                  (2,1): Protein→RNA
                  (2,2): Protein→Protein
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = _resolve_runtime_device(device)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        if isinstance(mode, str):
            self.mode = eval(mode)  # 将字符串转换为元组
        else:
            self.mode = mode



        self.in_dim = input_dim1 if self.mode[0] == 1 else input_dim2
        self.out_dim = input_dim1 if self.mode[1] == 1 else input_dim2
          # 加载模型
        self._load_model(model_path)
        
        print(f"✓ 传输映射初始化成功")
        print(f"  模式: {self.mode[0]}→{self.mode[1]}")
        print(f"  输入维度: {self.in_dim}, 输出维度: {self.out_dim}")
    
    def _load_model(self, model_path: str):
        """加载模型"""
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        self.model = SplicedAutoEncoder(
            input_dim1=self.input_dim1,
            input_dim2=self.input_dim2,
            hidden_dim=self.hidden_dim,
            flat_mode=True
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def T(self, x: torch.Tensor) -> torch.Tensor:
        """
        传输映射函数 T(x)
        
        参数:
            x: 输入张量 (batch_size, in_dim)
        返回:
            输出张量 (batch_size, out_dim)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        with torch.no_grad():
            output, _ = self.model.forward_single(
                x,
                in_domain=self.mode[0],
                out_domain=self.mode[1]
            )
        
        return output
    
    def T_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        传输映射函数 T(x)
        
        参数:
            x: 输入张量 (batch_size, in_dim)
        返回:
            输出张量 (batch_size, out_dim)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        with torch.no_grad():
            _, encoded = self.model.forward_single(
                x,
                in_domain=self.mode[0],
                out_domain=self.mode[1]
            )
        
        return encoded

    def T_rev(self, x: torch.Tensor) -> torch.Tensor:
        """
        传输映射函数 T(x)
        
        参数:
            x: 输入张量 (batch_size, in_dim)
        返回:
            输出张量 (batch_size, out_dim)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        with torch.no_grad():
            output, _ = self.model.forward_single(
                x,
                in_domain=self.mode[1],
                out_domain=self.mode[0]
            )
        
        return output

    def grad_T(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算雅可比矩阵 ∂T/∂x
        
        参数:
            x: 输入张量 (batch_size, in_dim)
        
        返回:
            雅可比矩阵 (batch_size, out_dim, in_dim)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # 前向传播
        output, _ = self.model.forward_single(
            x,
            in_domain=self.mode[0],
            out_domain=self.mode[1]
        )
        
        batch_size = x.shape[0]
        jacobian = torch.zeros(batch_size, self.out_dim, self.in_dim, device=self.device)
        
        # 对每个输出维度计算梯度
        for i in range(self.out_dim):
            # 创建梯度向量（只有第i个位置为1）
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1.0
            
            # 计算梯度
            if i == 0:
                # 第一次需要创建计算图
                grad_input = torch.autograd.grad(
                    outputs=output,
                    inputs=x,
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=False
                )[0]
            else:
                grad_input = torch.autograd.grad(
                    outputs=output,
                    inputs=x,
                    grad_outputs=grad_output,
                    retain_graph=(i < self.out_dim - 1),
                    create_graph=False
                )[0]
            
            jacobian[:, i, :] = grad_input
        
        return jacobian
    def sec_vecolity(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用雅可比矩阵计算次空间能量（基于论文公式）
        
        根据公式: energy = α * v^T (∇T)^T (∇T) v
        
        参数:
            x: 基准点 (batch_size, in_dim)
            v: 速度场 (batch_size, in_dim)
            alpha: 权重系数
        
        返回:
            能量 (batch_size, 1)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if not isinstance(v, torch.Tensor):
            v = torch.FloatTensor(v)
        v=v.unsqueeze(2)   # v: first_velocity (batch_size, in_dim, 1)

        
        x = x.to(self.device)
        v = v.to(self.device)

       

        jac = self.grad_T(x)     # jac matrix (batch_size, out_dim, in_dim)
        jacv = torch.bmm(jac, v) # jac * V    (size, out_dim, 1)
        jacv=jacv.squeeze(2)    

        return jacv

    def energy_Secondary(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        使用雅可比矩阵计算次空间能量（基于论文公式）
        
        根据公式: energy = α * v^T (∇T)^T (∇T) v
        
        参数:
            x: 基准点 (batch_size, in_dim)
            v: 速度场 (batch_size, in_dim)
            alpha: 权重系数
        
        返回:
            能量 (batch_size, 1)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if not isinstance(v, torch.Tensor):
            v = torch.FloatTensor(v)
        v=v.unsqueeze(2)
        
        x = x.to(self.device)
        v = v.to(self.device)
        
        # 计算雅可比矩阵 ∇T: (batch_size, out_dim, in_dim)
        jac = self.grad_T(x)
        jacv = torch.bmm(jac, v)  # 结果形状为 (size, dim2, 1)
        jacv=jacv.squeeze(2)     # 结果形状为 (size, dim2)

        energy = torch.sum(jacv ** 2, dim=1, keepdim=True)  # 形状为 (size, 1)
        
        return alpha * energy

class TransportMapFactory:
    """传输映射工厂类"""
    
    @staticmethod
    def create_all_maps(
        model_path: str,
        input_dim1: int,
        input_dim2: int,
        hidden_dim: int = 16,
        device: str = 'cpu'
    ) -> dict:
        """
        创建所有四种传输映射
        
        参数:
            model_path: 模型路径
            input_dim1: RNA 潜在维度
            input_dim2: Protein 潜在维度
            hidden_dim: 隐藏层维度
            device: 计算设备
        
        返回:
            包含四个映射的字典 {'T11': ..., 'T12': ..., 'T21': ..., 'T22': ...}
        """
        maps = {
            'T11': TransportMap(
                model_path, input_dim1, input_dim2, mode=(1, 1),
                hidden_dim=hidden_dim, device=device
            ),
            'T12': TransportMap(
                model_path, input_dim1, input_dim2, mode=(1, 2),
                hidden_dim=hidden_dim, device=device
            ),
            'T21': TransportMap(
                model_path, input_dim1, input_dim2, mode=(2, 1),
                hidden_dim=hidden_dim, device=device
            ),
            'T22': TransportMap(
                model_path, input_dim1, input_dim2, mode=(2, 2),
                hidden_dim=hidden_dim, device=device
            )
        }
        
        print("\n✓ 所有传输映射创建完成")
        return maps
