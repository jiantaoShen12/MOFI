import json
import pathlib
import shutil
from typing import Dict, Any
import scanpy as sc
import torch
import yaml
from CytoBridge.utils.config import load_config
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.trainer import TrainingPipeline
from CytoBridge.tl.interaction import cal_interaction
from scipy.sparse import issparse

def fit(adata: sc.AnnData,
        config: Dict[str, Any] | str,
        batch_size: int | None = None,
        adata_sec: sc.AnnData | None = None ,
        device: str = 'cuda',
        progress_callback=None) -> sc.AnnData:
    """

    TODO: add hold-out split
    TODO: save training curves / checkpoints to log directory
    TODO: support resume from checkpoint
    """
    # ---------- 1. load & resolve config ----------
    resolved_config = load_config(config)  # allow str-path or dict
    device = torch.device(device)

    # ---------- 2. data preparation ----------
    time_key = 'time_point_processed'
    time_points = sorted(adata.obs[time_key].unique())
    data_torch = []
    data_sec_torch = []
    hold_out = resolved_config["model"].get('hold_out', None)
    hold_out_sec = resolved_config["model"].get('hold_out_sec', None)

    if adata_sec is not None:
        # ---------- 2. data preparation ----------
        time_key = 'time_point_processed'
        time_points = sorted(adata_sec.obs[time_key].unique())

        if hold_out is not None:
            # 从 time_points 中移除所有 hold_out 的值
            for ho in hold_out:
                if ho in time_points:
                    time_points.remove(ho)
                    print(f"Sub space hold out time {ho}")

        for t in time_points:
            if hold_out_sec is not None and t in hold_out_sec:
                # 跳过该时间点，放入占位符（保持列表长度不变）
                # 用 None 作为标记，训练时判断 is None 即可
                data_sec_torch.append(None)
                print(f"Secondary space hold out time {t} (skipped)")
                continue  # 跳过后续数据构建
            
        # ======================================================
            subset = adata_sec[adata_sec.obs[time_key] == t]
                # 检测是否为稀疏矩阵
            if issparse(subset.obsm['X_latent']):
                # 如果是稀疏矩阵，转换为密集矩阵
                tens = torch.tensor(subset.obsm['X_latent'].toarray(), dtype=torch.float32, device=device)
            else:
                # 如果已经是密集矩阵，直接转换
                tens = torch.tensor(subset.obsm['X_latent'], dtype=torch.float32, device=device)

            data_sec_torch.append(tens)


    if hold_out is not None:
        for ho in hold_out:
            if ho in time_points:
                time_points.remove(ho)
                print(f"Primary space hold out time {ho}")
    for t in time_points:
        subset = adata[adata.obs[time_key] == t]
            # 检测是否为稀疏矩阵
        if issparse(subset.obsm['X_latent']):
            # 如果是稀疏矩阵，转换为密集矩阵
            tens = torch.tensor(subset.obsm['X_latent'].toarray(), dtype=torch.float32, device=device)
        else:
            # 如果已经是密集矩阵，直接转换
            tens = torch.tensor(subset.obsm['X_latent'], dtype=torch.float32, device=device)

        data_torch.append(tens)


    # ---------- 3. auto batch-size ----------
    if batch_size is None:
        batch_size = min(min(x.shape[0] for x in data_torch), 256)

    # ---------- 4. build & train model ----------
    dim = data_torch[0].shape[1]
    model = DynamicalModel(dim, resolved_config['model'])
    trainer = TrainingPipeline(model, resolved_config, batch_size, device, data=data_torch, progress_callback=progress_callback)
    print(trainer)
    if  data_sec_torch == []:
        model = trainer.train(data_torch, time_points)
    else :
        model = trainer.train(data_torch, time_points,data_sec_torch=data_sec_torch)


    # ---------- 5. compute latent outputs ----------
    all_times = torch.tensor(adata.obs[time_key].values, dtype=torch.float32, device=device).unsqueeze(1)
    all_data = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    net_input = torch.cat([all_data, all_times], dim=1)

    velocity = model.velocity_net(net_input)
    adata.obsm['velocity_latent'] = velocity.detach().cpu().numpy()

    if 'growth' in model.components:
        growth = model.growth_net(net_input)
        adata.obsm['growth_rate'] = growth.detach().cpu().numpy()

    if 'score' in model.components:
        score = model.score_net(net_input)
        adata.obsm['score_latent'] = score.detach().cpu().numpy()
    print(model.components)

    # ---------- 6. store model internals ----------
    adata.uns['all_model'] = {
        'model_config': resolved_config['model'],
        'training_config': {
            'defaults': resolved_config['training']['defaults'],
            'plan': json.dumps(resolved_config['training']['plan'])
        },
        'model_state_dict': {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    }

    # ---------- 7. save ----------
    ckpt_dir = pathlib.Path(resolved_config['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = ckpt_dir / 'adata.h5ad'
    yaml_path = ckpt_dir / 'config.yaml'

    adata.write_h5ad(h5ad_path)
    with yaml_path.open('w', encoding='utf-8') as yf:
        yaml.dump(resolved_config, yf, default_flow_style=False, allow_unicode=True)

    print(f"Model & data saved -> {ckpt_dir}")
    
    # metrics = trainer.evaluate(adata, data_torch, time_points)  # TODO handle hold-out
    metrics = {
            'w1_scores': 0.0,
            'tmv_scores': 0.0
        }
    adata.uns['evaluation_metrics'] = metrics
    print(adata)
    return adata

from CytoBridge.utils.utils import load_model_from_adata  



def fit_again(
    trained_adata: sc.AnnData,
    new_config: Dict[str, Any] | str,
    batch_size: int | None = None,
    adata_sec: sc.AnnData | None = None,
    device: str = 'cuda',
    progress_callback=None) -> sc.AnnData:
    """
    从已训练的adata中加载模型，使用新的配置继续训练，模型信息存储格式与fit函数保持一致
    """
    # ---------- 1. 加载并合并配置（旧plan在前，新plan在后） ----------
    # 加载新配置
    resolved_new_config = load_config(new_config)
    # 提取旧配置中的training plan
    old_training_config = trained_adata.uns['all_model']['training_config']
    old_plan = json.loads(old_training_config['plan'])  # 解析旧plan
    
    # 处理旧plan格式（确保为列表）
    old_plan = old_plan if isinstance(old_plan, list) else [old_plan]
    # 处理新plan格式
    new_plan = resolved_new_config['training'].get('plan', [])
    new_plan = new_plan if isinstance(new_plan, list) else [new_plan]
    # 合并plan：旧plan在前，新plan在后
    merged_plan = old_plan + new_plan
    device = torch.device(device)
    
    # ---------- 2. 数据准备（与fit函数保持一致） ----------
    data_sec_torch = []
    if adata_sec is not None:
        time_key = 'time_point_processed'
        time_points_sec = sorted(adata_sec.obs[time_key].unique())
        for t in time_points_sec:
            subset = adata_sec[adata_sec.obs[time_key] == t]
            if issparse(subset.obsm['X_latent']):
                tens = torch.tensor(subset.obsm['X_latent'].toarray(), dtype=torch.float32, device=device)
            else:
                tens = torch.tensor(subset.obsm['X_latent'], dtype=torch.float32, device=device)
            data_sec_torch.append(tens)

    time_key = 'time_point_processed'
    time_points = sorted(trained_adata.obs[time_key].unique())
    data_torch = []
    for t in time_points:
        subset = trained_adata[trained_adata.obs[time_key] == t]
        if issparse(subset.obsm['X_latent']):
            tens = torch.tensor(subset.obsm['X_latent'].toarray(), dtype=torch.float32, device=device)
        else:
            tens = torch.tensor(subset.obsm['X_latent'], dtype=torch.float32, device=device)
        data_torch.append(tens)
    
    # ---------- 3. 自动确定批处理大小 ----------
    if batch_size is None:
        batch_size = min(min(x.shape[0] for x in data_torch), 256)
    
    # ---------- 4. 加载已训练模型 ----------
    print("从已训练的adata中加载模型...")
    model = load_model_from_adata(trained_adata).to(device)
    
    # ---------- 5. 继续训练 ----------
    trainer = TrainingPipeline(model, resolved_new_config, batch_size, device, data=data_torch, progress_callback=progress_callback)
    print("开始继续训练...")
    if not data_sec_torch:
        model = trainer.train(data_torch, time_points)
    else:
        trainer.evaluate_map(trained_adata.obsm['X_latent'],adata_sec.obsm['X_latent']) 
        model = trainer.train(data_torch, time_points, data_sec_torch=data_sec_torch)

    # ---------- 6. 计算并更新潜在空间输出（与fit函数一致） ----------
    all_times = torch.tensor(
        trained_adata.obs[time_key].values, 
        dtype=torch.float32, 
        device=device
    ).unsqueeze(1)
    all_data = torch.tensor(
        trained_adata.obsm['X_latent'], 
        dtype=torch.float32, 
        device=device
    )
    net_input = torch.cat([all_data, all_times], dim=1)

    # 更新速度信息
    velocity = model.velocity_net(net_input)
    trained_adata.obsm['velocity_latent'] = velocity.detach().cpu().numpy()

    # 更新生长率信息
    if 'growth' in model.components:
        growth = model.growth_net(net_input)
        trained_adata.obsm['growth_rate'] = growth.detach().cpu().numpy()

    # 更新分数信息
    if 'score' in model.components:
        score = model.score_net(net_input)
        trained_adata.obsm['score_latent'] = score.detach().cpu().numpy()
    resolved_new_config['training']['plan'] = merged_plan  # 更新新配置的plan

    # ---------- 7. 存储模型信息（与fit函数格式完全一致） ----------
    trained_adata.uns['all_model'] = {
        'model_config': resolved_new_config['model'],  # 新模型配置
        'training_config': {
            'defaults': resolved_new_config['training']['defaults'],
            'plan': json.dumps(merged_plan)  # 合并后的plan序列化
        },
        'model_state_dict': {k: v.cpu().numpy() for k, v in model.state_dict().items()}  # 状态字典转CPU
    }

    # ---------- 8. 保存文件（与fit函数一致） ----------
    ckpt_dir = pathlib.Path(resolved_new_config['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 保存adata和配置文件
    h5ad_path = ckpt_dir / 'adata.h5ad'
    yaml_path = ckpt_dir / 'config.yaml'
    trained_adata.write_h5ad(h5ad_path)
    with yaml_path.open('w', encoding='utf-8') as yf:
        yaml.dump(resolved_new_config, yf, default_flow_style=False, allow_unicode=True)
    if trainer.tranmap is not None:
        map_model = getattr(trainer.tranmap, "model", None)
        can_save = (
            map_model is not None
            and hasattr(map_model, "state_dict")
            and bool(getattr(trainer.tranmap, "supports_training", True))
        )
        if can_save:
            torch.save(map_model.state_dict(), ckpt_dir / 'tranmap_bset_model.pt')
        else:
            print("skip tranmap checkpoint save: mapper has no trainable torch model")

    # 评估（与fit函数一致）
    trainer.evaluate(trained_adata, data_torch, time_points)

    if data_sec_torch:
        trainer.evaluate_map(trained_adata.obsm['X_latent'],adata_sec.obsm['X_latent']) 
    print(f"模型及数据已保存至 -> {ckpt_dir}")
    return trained_adata
