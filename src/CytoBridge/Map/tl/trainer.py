"""
训练脚本（遵循 BABEL 训练流程）
支持基础训练 + 交叉重建训练切换 + 早停策略
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn as nn

from CytoBridge.Map.tl.models import (
    SplicedAutoEncoder,
    QuadLoss,
    prepare_data,
    create_dataloaders
)

def train_epoch(model, dataloader, criterion, optimizer, device, use_cross_recon=False):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for x_batch in tqdm(dataloader, desc="Training"):
        x_batch = x_batch.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播（区分基础/交叉模式）
        if use_cross_recon:
            y_pred = model(x_batch, return_cross=True)
        else:
            y_pred = model(x_batch)
        
        # 分割真实值
        rna_dim = model.input_dim1
        rna_true = x_batch[:, :rna_dim]
        protein_true = x_batch[:, rna_dim:]
        y_true = (rna_true, protein_true)
        
        # 计算损失
        loss = criterion(y_pred, y_true)
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader), criterion.get_last_losses()

def validate(model, dataloader, criterion, device, use_cross_recon=False):
    """验证"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch in dataloader:
            x_batch = x_batch.to(device)
            
            # 前向传播（区分基础/交叉模式）
            if use_cross_recon:
                y_pred = model(x_batch, return_cross=True)
            else:
                y_pred = model(x_batch)
            
            rna_dim = model.input_dim1
            rna_true = x_batch[:, :rna_dim]
            protein_true = x_batch[:, rna_dim:]
            y_true = (rna_true, protein_true)

            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
    
    return total_loss / len(dataloader), criterion.get_last_losses()

def negative_binom_loss(
    scale_factor: float = 1.0,
    eps: float = 1e-10,
    mean: bool = True,
    debug: bool = False,
) -> callable:
    """
    Return a function that calculates the binomial loss
    https://github.com/theislab/dca/blob/master/dca/loss.py

    combination of the Poisson distribution and a gamma distribution is a negative binomial distribution
    """
    def loss(preds, theta, truth, tb_step: int = None):
        """Calculates negative binomial loss as defined in the NB class in link above"""
        y_true = truth
        y_pred = preds * scale_factor

        if debug:  # Sanity check before loss calculation
            assert not torch.isnan(y_pred).any(), y_pred
            assert not torch.isinf(y_pred).any(), y_pred
            assert not (y_pred < 0).any()  # should be non-negative
            assert not (theta < 0).any()

        # Clip theta values
        theta = torch.clamp(theta, max=1e6)

        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
        )
        t2 = (theta + y_true) * torch.log1p(y_pred / (theta + eps)) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )
        if debug:  # Sanity check after calculating loss
            assert not torch.isnan(t1).any(), t1
            assert not torch.isinf(t1).any(), (t1, torch.sum(torch.isinf(t1)))
            assert not torch.isnan(t2).any(), t2
            assert not torch.isinf(t2).any(), t2

        retval = t1 + t2
        if debug:
            assert not torch.isnan(retval).any(), retval
            assert not torch.isinf(retval).any(), retval

        return torch.mean(retval) if mean else retval

    return loss

class NegativeBinomialLoss(torch.nn.Module):
    """
    Negative binomial loss. Preds should be a tuple of (mean, dispersion)
    """
    def __init__(
        self,
        scale_factor: float = 1.0,
        eps: float = 1e-10,
        l1_lambda: float = 0.0,
        mean: bool = True,
    ):
        super(NegativeBinomialLoss, self).__init__()
        self.loss = negative_binom_loss(
            scale_factor=scale_factor,
            eps=eps,
            mean=mean,
            debug=True,
        )
        self.l1_lambda = l1_lambda

    def forward(self, preds, target):
        preds, theta = preds[:2]
        l = self.loss(
            preds=preds,
            theta=theta,
            truth=target,
        )
        encoded = preds[:-1]
        l += self.l1_lambda * torch.abs(encoded).sum()
        return l

def plot_loss_history(history, save_path, ext):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 总损失
    ax = axes[0, 0]
    ax.plot(history['train_total'], label='Train', linewidth=2)
    ax.plot(history['valid_total'], label='Valid', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 分项损失
    loss_names = ['loss_11', 'loss_22', 'loss_12', 'loss_21']
    titles = ['RNA→RNA', 'Protein→Protein', 'RNA→Protein', 'Protein→RNA']
    
    for idx, (loss_name, title) in enumerate(zip(loss_names, titles)):
        ax = axes.flat[idx + 1]
        ax.plot(history[f'train_{loss_name}'], label='Train', linewidth=2)
        ax.plot(history[f'valid_{loss_name}'], label='Valid', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 学习率
    ax = axes[1, 2]
    ax.plot(history['lr'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / f'loss.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------- 第二张图：Loss Z 单独可视化 --------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_title('Loss Z (Encoded Space Consistency)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Z Value')
    ax2.grid(True, alpha=0.3) 
    loss_name="loss_z"
    ax2.plot(history[f'train_{loss_name}'], label='Train', linewidth=2)
    ax2.plot(history[f'valid_{loss_name}'], label='Valid', linewidth=2)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path / f'loss_z.{ext}', dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {save_path / f'loss_z.{ext}'}")
    plt.close()

def plot_loss_history1(
    train_losses: list,  # 每个epoch的训练总损失列表
    val_losses: list,    # 每个epoch的验证总损失列表
    train_loss_details: list,  # 每个epoch的训练细分损失（含loss_z）
    val_loss_details: list = None,  # 验证细分损失（可选）
    save_path: str = "./loss_plots",
    figsize: tuple = (12, 10)
):
    """
    绘制损失曲线：
    - 图1：total loss + loss_11/loss_12/loss_21/loss_22（训练+验证）
    - 图2：loss_z（训练+验证）
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 准备epoch轴
    epochs = np.arange(1, len(train_losses) + 1)
    
    # 创建画布（2行1列）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Training & Validation Loss History', fontsize=14, fontweight='bold')

    # -------------------------- 第一张图：总损失 + 四路分支损失 --------------------------
    ax1.set_title('Total Loss & Branch Losses (11/12/21/22)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, alpha=0.3)

    # 1. 总损失
    ax1.plot(epochs, train_losses, 'b-', label='Train Total Loss', linewidth=2)
    if val_losses is not None and len(val_losses) > 0:
        ax1.plot(epochs, val_losses, 'r-', label='Val Total Loss', linewidth=2)
    
    # 2. 四路分支损失（训练）
    train_11 = [d['loss_11'] for d in train_loss_details]
    train_12 = [d['loss_12'] for d in train_loss_details]
    train_21 = [d['loss_21'] for d in train_loss_details]
    train_22 = [d['loss_22'] for d in train_loss_details]
    
    ax1.plot(epochs, train_11, 'b--', label='Train Loss 11', alpha=0.7)
    ax1.plot(epochs, train_12, 'g--', label='Train Loss 12', alpha=0.7)
    ax1.plot(epochs, train_21, 'y--', label='Train Loss 21', alpha=0.7)
    ax1.plot(epochs, train_22, 'c--', label='Train Loss 22', alpha=0.7)
    
    # 3. 四路分支损失（验证，可选）
    if val_loss_details is not None and len(val_loss_details) > 0:
        val_11 = [d['loss_11'] for d in val_loss_details]
        val_12 = [d['loss_12'] for d in val_loss_details]
        val_21 = [d['loss_21'] for d in val_loss_details]
        val_22 = [d['loss_22'] for d in val_loss_details]
        
        ax1.plot(epochs, val_11, 'r--', label='Val Loss 11', alpha=0.7)
        ax1.plot(epochs, val_12, 'm--', label='Val Loss 12', alpha=0.7)
        ax1.plot(epochs, val_21, 'k--', label='Val Loss 21', alpha=0.7)
        ax1.plot(epochs, val_22, 'orange', label='Val Loss 22', alpha=0.7)
    
    # 4. 交叉损失（如果有）
    if 'loss_121' in train_loss_details[0]:
        train_121 = [d['loss_121'] for d in train_loss_details]
        train_212 = [d['loss_212'] for d in train_loss_details]
        ax1.plot(epochs, train_121, 'b:', label='Train Loss 121 (1→2→1)', alpha=0.7)
        ax1.plot(epochs, train_212, 'g:', label='Train Loss 212 (2→1→2)', alpha=0.7)
        if val_loss_details is not None and len(val_loss_details) > 0:
            val_121 = [d['loss_121'] for d in val_loss_details]
            val_212 = [d['loss_212'] for d in val_loss_details]
            ax1.plot(epochs, val_121, 'r:', label='Val Loss 121 (1→2→1)', alpha=0.7)
            ax1.plot(epochs, val_212, 'm:', label='Val Loss 212 (2→1→2)', alpha=0.7)
    
    ax1.legend(loc='upper right', fontsize=8)
    ax1.tick_params(axis='both', labelsize=8)

    # -------------------------- 第二张图：Loss Z --------------------------
    ax2.set_title('Loss Z (Encoded Space Consistency)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Z Value')
    ax2.grid(True, alpha=0.3)
    
    train_z = [d['loss_z'] for d in train_loss_details]
    ax2.plot(epochs, train_z, 'b-', label='Train Loss Z', linewidth=2)
    if val_loss_details is not None and len(val_loss_details) > 0:
        val_z = [d['loss_z'] for d in val_loss_details]
        ax2.plot(epochs, val_z, 'r-', label='Val Loss Z', linewidth=2)
    
    ax2.legend(loc='upper right', fontsize=8)
    ax2.tick_params(axis='both', labelsize=8)

    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_complete.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='训练潜在空间转换模型')
        parser.add_argument('--main_adata_path', type=str, required=True)
        parser.add_argument('--sub_adata_path', type=str, required=True)
        parser.add_argument('--latent_key', type=str, default='X_latent')
        parser.add_argument('--hidden_dim', type=int, default=16)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        parser.add_argument('--save_path', type=str, default='./train_output')
        parser.add_argument('--patience_stage1', type=int, default=30, help='基础阶段早停步数')
        parser.add_argument('--patience_stage2', type=int, default=20, help='交叉阶段早停步数')
        parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
        parser.add_argument('--use_cross_recon', action='store_true', help='是否直接启用交叉重建')
        parser.add_argument('--auto_switch_cross', action='store_true', help='是否自动切换到交叉重建')
        parser.add_argument('--loss_plot_ext', type=str, default='png', help='损失图保存格式')
        
        args = parser.parse_args()
    
    # 创建保存目录
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n=== 加载数据 ===")
    rna_adata = sc.read_h5ad(args.main_adata_path)
    protein_adata = sc.read_h5ad(args.sub_adata_path)
    
    datasets = prepare_data(
        rna_adata, protein_adata,
        latent_key=args.latent_key,
        random_state=args.seed
    )
    
    dataloaders = create_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=int(getattr(args, "num_workers", 4)),
    )
    
    # 创建模型
    print("\n=== 创建模型 ===")
    rna_dim = datasets['train'].rna_latent.shape[1]
    print(rna_dim)
    protein_dim = datasets['train'].protein_latent.shape[1]
    print(protein_dim)

    
    # 初始化模型和损失函数
    model = SplicedAutoEncoder(
        input_dim1=rna_dim,
        input_dim2=protein_dim,
        hidden_dim=args.hidden_dim
    ).to(args.device)
    
    criterion = QuadLoss(
        losstype1='mse',
        losstype2='mse',
        loss11_weight=float(getattr(args, "lossweight11", 1.0)),
        loss22_weight=float(getattr(args, "lossweight22", 1.0)),
        loss12_weight=float(getattr(args, "lossweight12", 1.0)),
        loss21_weight=float(getattr(args, "lossweight21", 10.0)),
        lossz_weight=float(getattr(args, "lossweight_z", 10.0)),
        loss_recon_use=args.use_cross_recon  # 初始是否启用交叉损失
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练历史记录
    history = {
        'train_total': [],
        'valid_total': [],
        'train_loss_11': [],
        'train_loss_12': [],
        'train_loss_21': [],
        'train_loss_22': [],
        'train_loss_z': [],
        'valid_loss_11': [],
        'valid_loss_12': [],
        'valid_loss_21': [],
        'valid_loss_22': [],
        'valid_loss_z': [],
        'lr': [],
    }
    
    # 早停相关参数
    best_val_loss = float('inf')
    counter = 0
    stage = 1  # 1:基础阶段，2:交叉阶段
    use_cross = args.use_cross_recon
    
    # 开始训练
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | Stage {stage} | Cross Recon: {use_cross}")
        
        # 训练
        train_loss, train_loss_detail = train_epoch(
            model, dataloaders['train'], criterion, optimizer, args.device, use_cross
        )
        # 验证
        val_loss, val_loss_detail = validate(
            model, dataloaders['valid'], criterion, args.device, use_cross
        )

        scheduler.step(val_loss)
        
        # 记录历史
        history['train_total'].append(train_loss)
        history['valid_total'].append(val_loss)
        history['train_loss_11'].append(train_loss_detail.get('loss_11', 0))
        history['train_loss_12'].append(train_loss_detail.get('loss_12', 0))
        history['train_loss_21'].append(train_loss_detail.get('loss_21', 0))
        history['train_loss_22'].append(train_loss_detail.get('loss_22', 0))
        history['train_loss_z'].append(train_loss_detail.get('loss_z', 0))
        history['valid_loss_11'].append(val_loss_detail.get('loss_11', 0))
        history['valid_loss_12'].append(val_loss_detail.get('loss_12', 0))
        history['valid_loss_21'].append(val_loss_detail.get('loss_21', 0))
        history['valid_loss_22'].append(val_loss_detail.get('loss_22', 0))
        history['valid_loss_z'].append(val_loss_detail.get('loss_z', 0))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 打印进度
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Loss Z: Train={train_loss_detail.get('loss_z', 0):.4f} | Val={val_loss_detail.get('loss_z', 0):.4f}")

        if val_loss < best_val_loss - 1e-6 :
            best_val_loss = val_loss
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': val_loss,
            }, save_path / 'best_model.pt')
            print(f"  ✓ 最佳模型")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")

        # 阶段切换（自动切换交叉重建）
        if args.auto_switch_cross and stage == 1 and counter >= args.patience_stage1:
            print(f"Stage 1: {args.patience_stage1} epochs no improvement → switch to cross reconstruction")
            stage = 2
            use_cross = True
            criterion.loss_recon_use = True  # 启用交叉损失
            best_val_loss = val_loss  # 重置最优损失
            counter = 0  # 重置计数器
        
        # 早停（交叉阶段）
        if stage == 2 and counter >= args.patience_stage2:
            print(f"Stage 2: {args.patience_stage2} epochs no improvement → early stop")
            break
    
    # 保存训练历史
    with open(save_path / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # 绘制损失图
    plot_loss_history(history, save_path, args.loss_plot_ext)
    
    # 额外的详细损失图
    plot_loss_history1(
        train_losses=history['train_total'],
        val_losses=history['valid_total'],
        train_loss_details=[{
            'loss_11': history['train_loss_11'][i],
            'loss_12': history['train_loss_12'][i],
            'loss_21': history['train_loss_21'][i],
            'loss_22': history['train_loss_22'][i],
            'loss_z': history['train_loss_z'][i],
            'loss_121': train_loss_detail.get('loss_121', 0) if i >= len(history['train_total']) - len(train_loss_detail) else 0,
            'loss_212': train_loss_detail.get('loss_212', 0) if i >= len(history['train_total']) - len(train_loss_detail) else 0,
        } for i in range(len(history['train_total']))],
        val_loss_details=[{
            'loss_11': history['valid_loss_11'][i],
            'loss_12': history['valid_loss_12'][i],
            'loss_21': history['valid_loss_21'][i],
            'loss_22': history['valid_loss_22'][i],
            'loss_z': history['valid_loss_z'][i],
            'loss_121': val_loss_detail.get('loss_121', 0) if i >= len(history['valid_total']) - len(val_loss_detail) else 0,
            'loss_212': val_loss_detail.get('loss_212', 0) if i >= len(history['valid_total']) - len(val_loss_detail) else 0,
        } for i in range(len(history['valid_total']))],
        save_path=str(save_path)
    )
    
    print(f"Training completed! Results saved to {save_path}")

if __name__ == "__main__":
    main()
