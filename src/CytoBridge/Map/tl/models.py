"""
基于 BABEL 的潜在空间转换模型
严格遵循 BABEL 的 SplicedAutoEncoder 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import sparse
import scanpy as sc
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Union
import ot

# ==================== 基础模块（严格遵循 BABEL） ====================
class Encoder(nn.Module):
    """BABEL 的标准 Encoder，支持残差连接，最后输出前加一个线性变换"""
    def __init__(self, num_inputs: int, num_units: int = 32, activation=nn.PReLU, use_residual: bool = True):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.use_residual = use_residual

        self.encode1 = nn.Linear(self.num_inputs, 32)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = activation()

        self.encode2 = nn.Linear(32, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = activation()

        # 最后一个线性变换层
        self.final_linear = nn.Linear(self.num_units, self.num_units)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))

        if self.use_residual and self.num_inputs == self.num_units:
            x += residual  # 添加残差连接

        # 最后一个线性变换
        x = self.final_linear(x)
        return x

class Decoder(nn.Module):
    """BABEL 的标准 Decoder（简化为单输出），支持残差连接，最后输出前加一个线性变换"""
    def __init__(
        self, 
        num_outputs: int, 
        num_units: int = 32,
        activation=nn.PReLU,
        final_activation=None,
        use_residual: bool = True
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_units = num_units
        self.use_residual = use_residual

        self.decode1 = nn.Linear(self.num_units, num_units)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(num_units)
        self.act1 = activation()

        self.decode2 = nn.Linear(num_units, self.num_units)
        nn.init.xavier_uniform_(self.decode2.weight)

        #新加的
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = activation()

        # 最后一个线性变换层
        self.final_linear = nn.Linear(self.num_units, self.num_outputs)
        nn.init.xavier_uniform_(self.final_linear.weight)

        self.final_activation = final_activation

    def forward(self, x, size_factors=None):
        residual = x
        x = self.act1(self.bn1(self.decode1(x)))
        #x = self.decode2(x)
        #新加的
        x = self.act2(self.bn2(self.decode2(x)))

        if self.use_residual and self.num_units == self.num_outputs:
            x += residual  # 添加残差连接

        # 最后一个线性变换
        x = self.final_linear(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

# ==================== 主模型（SplicedAutoEncoder） ====================
class SplicedAutoEncoder(nn.Module):
    """
    基于 BABEL 的 SplicedAutoEncoder
    支持 RNA 和 Protein 潜在空间的四路转换 + 交叉重建（1→2→1 / 2→1→2）
    """
    def __init__(
        self,
        input_dim1: int,  # RNA 潜在维度
        input_dim2: int,  # Protein 潜在维度
        hidden_dim: int = 16,
        final_activations1=None,  # RNA decoder 激活
        final_activations2=None,  # Protein decoder 激活
        flat_mode: bool = True,
        seed: int = 42
    ):
        super().__init__()
        torch.manual_seed(seed)
        
        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        
        # 编码器
        self.encoder1 = Encoder(num_inputs=input_dim1, num_units=hidden_dim)
        self.encoder2 = Encoder(num_inputs=input_dim2, num_units=hidden_dim)
        
        # 解码器
        self.decoder1 = Decoder(
            num_outputs=input_dim1,
            num_units=hidden_dim,
            final_activation=final_activations1
        )
        self.decoder2 = Decoder(
            num_outputs=input_dim2,
            num_units=hidden_dim,
            final_activation=final_activations2
        )
    
    def _combine_output_and_encoded(self, decoded, encoded):
        """组合输出和编码"""
        return (decoded, encoded)
    
    def forward_single(
        self, x, size_factors=None, in_domain: int = 1, out_domain: int = 1
    ):
        """单路前向传播"""
        encoder = self.encoder1 if in_domain == 1 else self.encoder2
        decoder = self.decoder1 if out_domain == 1 else self.decoder2
        encoded = encoder(x)
        decoded = decoder(encoded, size_factors)
        return self._combine_output_and_encoded(decoded, encoded)
    
    def forward(self, x, size_factors=None, mode: Union[None, Tuple[int, int]] = None, return_cross: bool = False):
        """
        四路前向传播 + 交叉重建输出
        
        参数:
            x: tuple of (rna_latent, protein_latent) 或 flat tensor
            size_factors: 尺寸因子（用于解码器）
            mode: 指定输出模式 (in, out)
            return_cross: 是否返回交叉重建结果（1→2→1 / 2→1→2）
        
        返回:
            - 基础模式: (retval11, retval12, retval21, retval22)
            - 交叉模式: (基础返回值, cross121, cross212)
              cross121: (1→2→1解码结果, 中间编码)
              cross212: (2→1→2解码结果, 中间编码)
        """
        if self.flat_mode:
            # 如果是 flat 模式，需要分割输入
            assert isinstance(x, torch.Tensor)
            x1, x2 = torch.split(x, [self.input_dim1, self.input_dim2], dim=-1)
            x = (x1, x2)
        
        assert isinstance(x, (tuple, list)) and len(x) == 2
        x1, x2 = x
        
        # 基础编码
        encoded1 = self.encoder1(x1)
        encoded2 = self.encoder2(x2)
        
        # 基础解码（四路）
        decoded11 = self.decoder1(encoded1, size_factors)
        decoded12 = self.decoder2(encoded1, size_factors)
        decoded21 = self.decoder1(encoded2, size_factors)
        decoded22 = self.decoder2(encoded2, size_factors)
        
        # 基础返回值
        retval11 = self._combine_output_and_encoded(decoded11, encoded1)
        retval12 = self._combine_output_and_encoded(decoded12, encoded1)
        retval21 = self._combine_output_and_encoded(decoded21, encoded2)
        retval22 = self._combine_output_and_encoded(decoded22, encoded2)
        
        # 如果指定mode，直接返回对应值
        if mode is not None:
            retval_dict = {
                (1, 1): retval11,
                (1, 2): retval12,
                (2, 1): retval21,
                (2, 2): retval22,
            }
            return retval_dict[mode]
        
        # 计算交叉重建（1→2→1 / 2→1→2）
        cross121 = None
        cross212 = None
        if return_cross:
            # 1→2→1: x1→encoder1→decoder2→encoder2→decoder1
            x1_to_2 = self.decoder2(encoded1, size_factors)
            x1_to_2_encoded = self.encoder2(x1_to_2)
            x1_cross = self.decoder1(x1_to_2_encoded, size_factors)
            cross121 = self._combine_output_and_encoded(x1_cross, x1_to_2_encoded)
            
            # 2→1→2: x2→encoder2→decoder1→encoder1→decoder2
            x2_to_1 = self.decoder1(encoded2, size_factors)
            x2_to_1_encoded = self.encoder1(x2_to_1)
            x2_cross = self.decoder2(x2_to_1_encoded, size_factors)
            cross212 = self._combine_output_and_encoded(x2_cross, x2_to_1_encoded)
        
        if return_cross:
            return (retval11, retval12, retval21, retval22), cross121, cross212
        else:
            return retval11, retval12, retval21, retval22

# ==================== 损失函数 ====================
class QuadLoss(nn.Module):
    """
    四路损失函数（仅支持平衡OT：EMD/Sinkhorn + MSE）
    支持基础损失 + 交叉重建损失（1→2→1 / 2→1→2）
    """
    def __init__(
        self,
        losstype1: str = 'mse',      # 仅支持 'mse'/'ot_sinkhorn'/'ot_emd'
        losstype2: str = 'mse',      # 仅支持 'mse'/'ot_sinkhorn'/'ot_emd'
        loss11_weight: float = 1.0,
        loss12_weight: float = 1.0,
        loss21_weight: float = 10.0,
        loss22_weight: float = 10.0,
        lossz_weight: float = 10.0,
        loss121_weight: float = 1.0,  # 1→2→1 交叉损失权重
        loss212_weight: float = 1.0,  # 2→1→2 交叉损失权重
        ot_reg: float = 0.01,
        loss_recon_use: bool = False,  # 是否启用交叉重建损失
        record_history: bool = True
    ):
        super().__init__()
        self.losstype1 = losstype1
        self.losstype2 = losstype2
        self.ot_reg = ot_reg
        
        # 权重与基础损失
        self.mse_loss = nn.MSELoss()
        self.loss11_weight = loss11_weight
        self.loss12_weight = loss12_weight
        self.loss21_weight = loss21_weight
        self.loss22_weight = loss22_weight
        self.lossz_weight = lossz_weight
        self.loss121_weight = loss121_weight
        self.loss212_weight = loss212_weight

        self.record_history = record_history
        self.loss_history = [] if record_history else None
        self.loss_recon_use = loss_recon_use  # 控制是否启用交叉损失

    def _compute_balanced_ot(self, pred, target, ot_type):
        """仅计算平衡OT（EMD/Sinkhorn），修复设备+梯度问题"""
        device = pred.device
        dtype = pred.dtype
        
        # 1. 构造平衡权重（均匀分布）
        batch_size = pred.shape[0]
        a = np.ones(batch_size) / batch_size
        b = np.ones(batch_size) / batch_size
        
        # 2. 成本矩阵（欧式距离平方，保留梯度）
        M = torch.cdist(pred, target, p=2) ** 2
        M_np = M.detach().cpu().numpy()  # POT仅支持numpy
        
        # 3. 计算OT传输矩阵
        if ot_type == 'emd':
            pi_np = ot.emd(a, b, M_np)
        elif ot_type == 'sinkhorn':
            pi_np = ot.sinkhorn(a, b, M_np, reg=self.ot_reg)
        else:
            raise ValueError(f"仅支持emd/sinkhorn，输入：{ot_type}")
        
        # 4. 转回tensor并计算损失（对齐设备，复用原M的梯度）
        pi = torch.tensor(pi_np, dtype=dtype, device=device)
        ot_loss = torch.sum(pi * M)
        return ot_loss

    def _compute_loss(self, loss_type, pred, target):
        """统一计算MSE/平衡OT损失"""
        if loss_type == 'mse':
            return self.mse_loss(pred, target)
        elif loss_type == 'ot_emd':
            return self._compute_balanced_ot(pred, target, 'emd')
        elif loss_type == 'ot_sinkhorn':
            return self._compute_balanced_ot(pred, target, 'sinkhorn')
        else:
            raise ValueError(f"仅支持mse/ot_emd/ot_sinkhorn，输入：{loss_type}")

    def forward(self, y_pred, y_true):
        """
        前向计算
        - 基础模式: y_pred = (pred11, encoded1), (pred12, _), (pred21, encoded2), (pred22, _)
        - 交叉模式: y_pred = ((基础4个), cross121, cross212)
        y_true: (rna_true, protein_true)
        """
        true1, true2 = y_true
        loss_121 = 0.0
        loss_212 = 0.0

        # 解析预测值（区分基础/交叉模式）
        if self.loss_recon_use and isinstance(y_pred, tuple) and len(y_pred) == 3:
            base_pred, cross121, cross212 = y_pred
            (pred11, encoded1), (pred12, _), (pred21, encoded2), (pred22, _) = base_pred
            # 提取交叉重建结果
            pred121, _ = cross121
            pred212, _ = cross212
        else:
            (pred11, encoded1), (pred12, _), (pred21, encoded2), (pred22, _) = y_pred

        # 计算基础分支损失
        loss_11 = self._compute_loss(self.losstype1, pred11, true1)
        loss_22 = self._compute_loss(self.losstype2, pred22, true2)
        loss_12 = self._compute_loss(self.losstype2, pred12, true2)
        loss_21 = self._compute_loss(self.losstype1, pred21, true1)
        loss_z = self._compute_loss(self.losstype1, encoded1, encoded2)

        # 计算交叉重建损失（如果启用）
        if self.loss_recon_use:
            loss_121 = self._compute_loss(self.losstype1, pred121, true1)  # 1→2→1
            loss_212 = self._compute_loss(self.losstype2, pred212, true2)  # 2→1→2

        # 总损失计算（区分基础/交叉模式）
        if not self.loss_recon_use:
            # 基础模式: (11,12,21,22,lossz)
            total_loss = (
                self.loss11_weight * loss_11 +
                self.loss22_weight * loss_22 +
                self.loss12_weight * loss_12 +
                self.loss21_weight * loss_21 +
                self.lossz_weight * loss_z
            )
        else:
            # 交叉模式: (12,121,21,212,lossz)
            total_loss = (
                self.loss12_weight * loss_12 +
                self.loss121_weight * loss_121 +
                self.loss21_weight * loss_21 +
                self.loss212_weight * loss_212 +
                self.lossz_weight * loss_z
            )

        # 记录损失
        if self.record_history:
            loss_record = {
                'total': total_loss.item(),
                'loss_11': loss_11.item(),
                'loss_12': loss_12.item(),
                'loss_21': loss_21.item(),
                'loss_22': loss_22.item(),
                'loss_z': loss_z.item(),
                'loss_121': loss_121.item() if self.loss_recon_use else 0.0,
                'loss_212': loss_212.item() if self.loss_recon_use else 0.0,
            }
            self.loss_history.append(loss_record)
        
        return total_loss

    def get_last_losses(self):
        """获取最后一次损失记录"""
        return self.loss_history[-1] if self.loss_history else {}

class QuadLoss_mse(nn.Module):
    """
    四路损失函数（遵循 BABEL）
    用法：
        loss_fn = QuadLoss(
            loss1='mse',          # 或 'ot'
            loss2='ot',
            loss11_weight=1.0,
            loss12_weight=0.8,
            loss21_weight=0.8,
            loss22_weight=1.0
        )
    """
    def __init__(
        self,
        losstype1: str = 'mse',      # 'mse' 或 'ot'
        losstype2: str = 'mse',      # 'mse' 或 'ot'
        loss11_weight: float = 1.0,
        loss12_weight: float = 1.0,
        loss21_weight: float = 10.0,
        loss22_weight: float = 10.0,
        lossz_weight: float = 10.0,
        record_history: bool = True
    ):
        super().__init__()
        self.loss1 = self._build_loss(losstype1)
        self.loss2 = self._build_loss(losstype2)

        self.loss11_weight = loss11_weight
        self.loss12_weight = loss12_weight
        self.loss21_weight = loss21_weight
        self.loss22_weight = loss22_weight
        self.lossz_weight = lossz_weight

        self.record_history = record_history
        if self.record_history:
            self.loss_history = []

    def _build_loss(self, loss_type: str):
        """根据字符串返回损失函数"""
        if loss_type == 'mse':
            return nn.MSELoss()
        if loss_type == 'ot':
            return 'ot'          # 占位，真正计算时走 OT 分支
        raise ValueError("loss_type 必须是 'mse' 或 'ot'")

    def _compute_loss(self, loss_fn, pred, target):
        """按需调用 MSE 或 OT"""
        if loss_fn == 'ot':
            B = pred.size(0)
            x = pred.view(B, -1)
            y = target.view(B, -1)
            C = torch.cdist(x, y, p=2) ** 2          # 成本矩阵
            a = torch.ones(B, device=pred.device) / B
            b = torch.ones(B, device=pred.device) / B
            T = ot.sinkhorn(a, b, C, reg=0.1)
            return torch.sum(T * C)
        else:  # MSE
            return loss_fn(pred, target)

    def forward(self, y_pred, y_true):
        (pred11, encoded1), (pred12, _), (pred21, encoded2), (pred22, _) = y_pred
        true1, true2 = y_true

        loss_11 = self._compute_loss(self.loss1, pred11, true1)
        loss_22 = self._compute_loss(self.loss2, pred22, true2)
        loss_12 = self._compute_loss(self.loss2, pred12, true2)
        loss_21 = self._compute_loss(self.loss1, pred21, true1)
        loss_z = self._compute_loss(self.loss1, encoded1, encoded2)

        total = (self.loss11_weight * loss_11 +
                 self.loss22_weight * loss_22 +
                 self.loss12_weight * loss_12 +
                 self.loss21_weight * loss_21 +
                 self.lossz_weight * loss_z )

        if self.record_history:
            self.loss_history.append({
                'total': total.item(),
                'loss_11': loss_11.item(),
                'loss_12': loss_12.item(),
                'loss_21': loss_21.item(),
                'loss_22': loss_22.item(),
                'loss_z': loss_z.item(),
            })
        return total

    def get_last_losses(self):
        return self.loss_history[-1] if self.loss_history else {}

class QuadLoss_ori(nn.Module):
    """
    四路损失函数（遵循 BABEL）
    """
    def __init__(
        self,
        loss1=None,  # 模态1 (RNA) 的损失
        loss2=None,  # 模态2 (Protein) 的损失
        loss2_weight: float = 1.0,  # 交叉转换损失权重
        record_history: bool = True
    ):
        super().__init__()
        self.loss1 = loss1 if loss1 is not None else nn.MSELoss()
        self.loss2 = loss2 if loss2 is not None else nn.MSELoss()
        self.loss2_weight = loss2_weight
        self.record_history = record_history
        
        if self.record_history:
            self.loss_history = []
    
    def forward(self, y_pred, y_true):
        """
        计算损失
        
        参数:
            y_pred: (out11, out12, out21, out22)
                    每个 out 是 (decoded, encoded)
            y_true: (true1, true2)
        """
        (pred11, enc1_a), (pred12, enc1_b), \
        (pred21, enc2_a), (pred22, enc2_b) = y_pred
        
        true1, true2 = y_true
        
        # 四个重建损失
        loss_11 = self.loss1(pred11, true1)  # 1→1
        loss_22 = self.loss2(pred22, true2)  # 2→2
        loss_12 = self.loss2(pred12, true2)  # 1→2 (交叉)
        loss_21 = self.loss1(pred21, true1)  # 2→1 (交叉)
        
        # 总损失
        total_loss = loss_11 + loss_22 + self.loss2_weight * (loss_12 + loss_21)
        
        if self.record_history:
            self.loss_history.append({
                'total': total_loss.item(),
                'loss_11': loss_11.item(),
                'loss_12': loss_12.item(),
                'loss_21': loss_21.item(),
                'loss_22': loss_22.item()
            })
        
        return total_loss
    
    def get_last_losses(self):
        return self.loss_history[-1] if self.loss_history else {}

class PairedLatentDataset(Dataset):
    """配对的潜在空间数据集"""
    def __init__(self, rna_latent, protein_latent, flat_mode=True):
        assert rna_latent.shape[0] == protein_latent.shape[0]
        self.rna_latent = rna_latent.astype(np.float32)
        self.protein_latent = protein_latent.astype(np.float32)
        self.flat_mode = flat_mode
    
    def __len__(self):
        return len(self.rna_latent)
    
    def __getitem__(self, idx):
        rna = self.rna_latent[idx]
        protein = self.protein_latent[idx]
        
        if self.flat_mode:
            # 拼接成单个向量
            x = np.concatenate([rna, protein])
            return x  # y=None 因为输入即输出
        else:
            return (rna, protein)
def prepare_data(
    rna_adata: sc.AnnData,
    protein_adata: sc.AnnData,
    latent_key: str = 'X_latent',
    test_size: float = 0.05,
    valid_size: float = 0.05,
    random_state: int = 42
) -> Dict:
    """准备数据集"""
    print(rna_adata)
    print(protein_adata)
    if latent_key ==None:
        rna_latent = rna_adata.X
        protein_latent = protein_adata.X
    else :
        rna_latent = rna_adata.obsm[latent_key]
        protein_latent = protein_adata.obsm[latent_key]
    
    assert rna_latent.shape[0] == protein_latent.shape[0]
    
    indices = np.arange(rna_latent.shape[0])
    train_valid_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    valid_ratio = valid_size / (1 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=valid_ratio, random_state=random_state
    )
    
    train_dataset = PairedLatentDataset(
        rna_latent[train_idx], protein_latent[train_idx], flat_mode=True
    )
    valid_dataset = PairedLatentDataset(
        rna_latent[valid_idx], protein_latent[valid_idx], flat_mode=True
    )
    test_dataset = PairedLatentDataset(
        rna_latent[test_idx], protein_latent[test_idx], flat_mode=True
    )

    print(f"数据: Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)}")
    print(f"维度: RNA={rna_latent.shape[1]}, Protein={protein_latent.shape[1]}")
    
    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset,
        'indices': {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    }

def create_dataloaders(datasets: Dict, batch_size: int = 128, 
                      num_workers: int = 4) -> Dict[str, DataLoader]:
    """创建 DataLoader"""
    return {
        'train': DataLoader(
            datasets['train'], batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=False
        ),
        'valid': DataLoader(
            datasets['valid'], batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=False
        ),
        'test': DataLoader(
            datasets['test'], batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=False
        )
    }