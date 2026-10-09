
import torch  
import numpy as np  
import scanpy as sc  
import matplotlib.pyplot as plt  
import pickle  
from pathlib import Path  
from typing import Tuple, Optional, Union, Dict, Any  
from umap import UMAP  
from CytoBridge.Map.tl.models import SplicedAutoEncoder  
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine


def _resolve_runtime_device(device: str) -> torch.device:
    requested = str(device).strip() or "cpu"
    if requested.lower().startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


class LatentTranslator:  
    """潜在空间转换器（推理）"""  
      
    def __init__(self, model_path: str, device: str = 'cpu'):  
        """  
        初始化  
          
        参数:  
            model_path: 训练好的模型路径  
            device: 计算设备  
        """  
        self.device = _resolve_runtime_device(device)  
          
        # 加载模型检查点  
        checkpoint = torch.load(model_path, map_location=self.device)  
        self.checkpoint = checkpoint  
        self.model = None  
          
    def load_model(self, input_dim1: int, input_dim2: int, hidden_dim: int = 16):  
        """加载模型"""  
        print(            "input_dim1",input_dim1,  
            "input_dim2",input_dim2,  
            "hidden_dim",hidden_dim,  )
        self.model = SplicedAutoEncoder(  
            input_dim1=input_dim1,  
            input_dim2=input_dim2,  
            hidden_dim=hidden_dim,  
            flat_mode=True  
        ).to(self.device)  
          
        # 加载权重  
        if 'model_state_dict' in self.checkpoint:  
            self.model.load_state_dict(self.checkpoint['model_state_dict'])  
        else:  
            self.model.load_state_dict(self.checkpoint)  
          
        self.model.eval()  
        print(f"✓ 模型加载成功")  
        print(f"  参数量: {sum(p.numel() for p in self.model.parameters()):,}")  
      
    def translate(  
        self,  
        data: np.ndarray,  
        mode: Tuple[int, int]  
    ) -> np.ndarray:  
        """  
        执行转换  
          
        参数:  
            data: 输入数据  
            mode: 转换模式 (in_domain, out_domain)  
          
        返回:  
            转换后的数据  
        """  
        with torch.no_grad():  
            data_tensor = torch.FloatTensor(data).to(self.device)  
            output, encoded = self.model.forward_single(  
                data_tensor,  
                in_domain=mode[0],  
                out_domain=mode[1]  
            )  
            return output.cpu().numpy()  

    def translate_encode(  
        self,  
        data: np.ndarray,  
        mode: Tuple[int, int]  
    ) -> np.ndarray:  
        """  
        执行转换  
          
        参数:  
            data: 输入数据  
            mode: 转换模式 (in_domain, out_domain)  
          
        返回:  
            转换后的数据  
        """  
        with torch.no_grad():  
            data_tensor = torch.FloatTensor(data).to(self.device)  
            _, output = self.model.forward_single(  
                data_tensor,  
                in_domain=mode[0],  
                out_domain=mode[1]  
            )  
            return output.cpu().numpy()


def plot_comparison(    
    original_data: np.ndarray,    
    translated_data: np.ndarray,    
    labels: np.ndarray = None,    
    label_name: str = "cell_type",
    mode_str: str = "Translation",    
    save_path: str = None,    
    figsize: Tuple[int, int] = (12, 5),    
    dim_reduction: str = 'umap',    
    input_umap_model: Optional[UMAP] = None,
    target_umap_model: Optional[UMAP] = None,

    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.5,
    umap_random_state: int = 42
):    
    """    
    绘制原始数据和转换后数据的对比图    
    - 如果dim_reduction为'none'，直接使用数据的前两个维度    
    - 如果dim_reduction为'umap'，使用UMAP模型进行降维（预训练模型投影）
    - 如果dim_reduction为'umap_independent'，对每个数据独立进行UMAP降维
    """    
    fig, axes = plt.subplots(1, 2, figsize=figsize)    
    
    # 根据降维方式处理数据
    if dim_reduction == 'none':    
        original_embedding = original_data[:, :2]    
        translated_embedding = translated_data[:, :2]    
    elif dim_reduction == 'umap':    
        if (input_umap_model is None) and (target_umap_model is None) :    
            raise ValueError("UMAP模型未提供，无法进行UMAP降维")    
        original_embedding = input_umap_model.transform(original_data)
        translated_embedding = target_umap_model.transform(translated_data)
    elif dim_reduction == 'umap_independent':
        # 对原始数据独立进行UMAP降维
        adata_orig_temp = sc.AnnData(original_data)
        sc.pp.neighbors(adata_orig_temp, n_neighbors=umap_n_neighbors, random_state=umap_random_state)
        sc.tl.umap(adata_orig_temp, min_dist=umap_min_dist, random_state=umap_random_state)
        original_embedding = adata_orig_temp.obsm['X_umap']
        
        # 对转换后数据独立进行UMAP降维
        adata_trans_temp = sc.AnnData(translated_data)
        sc.pp.neighbors(adata_trans_temp, n_neighbors=umap_n_neighbors, random_state=umap_random_state)
        sc.tl.umap(adata_trans_temp, min_dist=umap_min_dist, random_state=umap_random_state)
        translated_embedding = adata_trans_temp.obsm['X_umap']
    else:    
        raise ValueError(f"不支持的降维方式: {dim_reduction}")    
    
    # 创建AnnData对象用于绘图
    adata_orig = sc.AnnData(original_data)    
    adata_trans = sc.AnnData(translated_data)    
    
    if labels is not None:
        label_name = str(label_name or "label")
        adata_orig.obs[label_name] = labels
        adata_trans.obs[label_name] = labels
    
    adata_orig.obsm['X_dimred'] = original_embedding    
    adata_trans.obsm['X_dimred'] = translated_embedding    
    
    color = label_name if labels is not None else None
    
    # 绘制    
    sc.pl.embedding(adata_orig, basis='X_dimred', color=color, ax=axes[0], show=False, title='Original')    
    sc.pl.embedding(adata_trans, basis='X_dimred', color=color, ax=axes[1], show=False, title=f'Translated ({mode_str})')    
    
    plt.suptitle(f'{mode_str} Comparison', fontsize=14, fontweight='bold', y=1.02)    
    plt.tight_layout()    
    
    if save_path:    
        plt.savefig(save_path, dpi=300, bbox_inches='tight')    
        print(f"✓ 图片已保存: {save_path}")    
    
    plt.close()    


def train_and_save_umap_model(  
    data: np.ndarray,  
    save_path: str,  
    n_neighbors: int = 15,  
    min_dist: float = 0.5,  
    random_state: int = 42,  
    n_components: int = 2  
) -> Tuple[UMAP, np.ndarray]:  
    """  
    训练UMAP模型并保存（用于直接投影）  
      
    参数:  
        data: 用于训练UMAP的数据（目标域原始数据）  
        save_path: UMAP模型保存路径  
        n_neighbors: UMAP邻居数量  
        min_dist: UMAP最小距离  
        random_state: 随机种子  
        n_components: UMAP输出维度  
      
    返回:  
        训练好的UMAP模型和原始数据的UMAP嵌入  
    """  
    # 训练UMAP模型  
    umap_model = UMAP(  
        n_neighbors=n_neighbors,  
        min_dist=min_dist,  
        random_state=random_state,  
        n_components=n_components,  
        verbose=False  
    )  
    umap_embedding = umap_model.fit_transform(data)  
    # 保存UMAP模型和嵌入  
    umap_info = {  
        'model': umap_model,  
        'embedding': umap_embedding,  
        'params': {  
            'n_neighbors': n_neighbors,  
            'min_dist': min_dist,  
            'random_state': random_state,  
            'n_components': n_components  
        }  
    }  
      
    with open(save_path, 'wb') as f:  
        pickle.dump(umap_info, f)  
      
    print(f"✓ UMAP模型已保存: {save_path}")  
    return umap_model, umap_embedding  


def load_umap_model(load_path: str) -> Dict[str, Any]:  
    """  
    加载保存的UMAP模型  
      
    参数:  
        load_path: UMAP模型加载路径  
      
    返回:  
        包含UMAP模型、嵌入和参数的字典  
    """  
    with open(load_path, 'rb') as f:  
        umap_info = pickle.load(f)  
      
    print(f"✓ UMAP模型已加载: {load_path}")  
    return umap_info  

def calculate_metrics(original: np.ndarray, recovered: np.ndarray, name: str) -> Dict[str, float]:
    """计算量化指标"""
    mse = mean_squared_error(original, recovered)
    nmse = mse / np.var(original) if np.var(original) > 1e-8 else mse
    cos_sim = np.mean([1 - cosine(o, r) for o, r in zip(original, recovered)])
    mae = np.mean(np.abs(original - recovered))
    rel_error = mae / (np.mean(np.abs(original)) + 1e-8)

    metrics = {
        f'{name}_mse': mse,
        f'{name}_nmse': nmse,
        f'{name}_cosine_similarity': cos_sim,
        f'{name}_mae': mae,
        f'{name}_relative_error': rel_error
    }

    # 打印
    print(f"\n  {name} 指标:")
    print(f"    MSE: {mse:.6f} | NMSE: {nmse:.6f}")
    print(f"    余弦相似度: {cos_sim:.4f} (理想=1)")
    print(f"    MAE: {mae:.6f} | 相对误差: {rel_error:.4f} (理想=0)")

    return metrics


def translate_single_mode(    
    model_path: str,    
    rna_adata_path: str,    
    protein_adata_path: str,    
    mode: str,    
    latent_key: str = 'X_latent',    
    hidden_dim: int = 16,    
    outdir: str = './inference_output',    
    cell_type_key: str = 'cell_type',    
    device: str = 'cpu',    
    dim_reduction: str = 'umap',
    translate_method: str = 'decode',
    umap_n_neighbors: int = 15,    
    umap_min_dist: float = 0.5,    
    umap_random_state: int = 42,
    model_state_path: Optional[str] = None
) -> Dict[str, Any]:    
    """    
    执行单个转换模式    
    - 1to1 (RNA→RNA): 转换后数据使用RNA的UMAP模型投影    
    - 1to2 (RNA→Protein): 转换后数据使用Protein的UMAP模型投影    
    - 2to1 (Protein→RNA): 转换后数据使用RNA的UMAP模型投影    
    - 2to2 (Protein→Protein): 转换后数据使用Protein的UMAP模型投影    
    
    参数:
        dim_reduction: 降维方式 ('none', 'umap', 'umap_independent')
            - 'none': 直接使用前两个维度
            - 'umap': 使用预训练UMAP模型进行投影
            - 'umap_independent': 对原始数据和转换数据分别独立进行UMAP降维
    """    
    outdir = Path(outdir)    
    outdir.mkdir(parents=True, exist_ok=True)    
    
    # UMAP模型保存路径（保存在根目录，所有模式共享）    
    root_outdir = outdir.parent if mode in ['1to1', '1to2', '2to1', '2to2'] else outdir    
    rna_umap_path = root_outdir / 'rna_umap_model.pkl'    
    protein_umap_path = root_outdir / 'protein_umap_model.pkl'    
    print("是否存在rna_umap_path：：",rna_umap_path.exists())
    print("是否存在protein_umap_path：：",protein_umap_path.exists())

    # 模式映射：(in_domain, out_domain, mode_name, input_domain, target_domain)    
    mode_map = {    
        '1to1': ((1, 1), 'RNA→RNA', 'rna', 'rna'),    
        '1to2': ((1, 2), 'RNA→Protein', 'rna', 'protein'),    
        '2to1': ((2, 1), 'Protein→RNA', 'protein', 'rna'),    
        '2to2': ((2, 2), 'Protein→Protein', 'protein', 'protein')    
    }    
    if mode not in mode_map:    
        raise ValueError(f"不支持的模式: {mode}。支持的模式: {list(mode_map.keys())}")    
    
    mode_tuple, mode_name, input_domain, target_domain = mode_map[mode]    
    
    print(f"\n{'='*60}")    
    print(f"转换模式: {mode_name}")    
    print(f"  输入域: {input_domain.upper()}, 目标域: {target_domain.upper()}")
    print(f"  降维方式: {dim_reduction}")
    print(f"{'='*60}")    
    
    # 加载数据    
    print("加载数据...")    
    rna_adata = sc.read_h5ad(rna_adata_path)    
    protein_adata = sc.read_h5ad(protein_adata_path)    
    
    if latent_key is None:    
        rna_latent = rna_adata.X    
        protein_latent = protein_adata.X    
    else:    
        rna_latent = rna_adata.obsm[latent_key]    
        protein_latent = protein_adata.obsm[latent_key]    
    
    print(f"  RNA 潜在维度: {rna_latent.shape}")    
    print(f"  Protein 潜在维度: {protein_latent.shape}")    
    
    # 获取标签（优先使用输入域的标签）    
    labels = None    
    if input_domain == 'rna' and cell_type_key in rna_adata.obs:    
        labels = rna_adata.obs[cell_type_key].values    
    elif input_domain == 'protein' and cell_type_key in protein_adata.obs:    
        labels = protein_adata.obs[cell_type_key].values    
    
    if labels is not None:    
        print(f"  细胞类型: {len(np.unique(labels))} 类")    
    else:    
        print("  警告: 未找到细胞类型标签，UMAP图将不进行着色")    
    
    # 处理降维模型    
    print("\n处理降维模型...")    
    umap_model = None    
    
    if dim_reduction == 'umap':    
        # 原有的UMAP投影逻辑
        print("input_domain",input_domain)
        if input_domain == 'rna':    
            if rna_umap_path.exists():    
                input_umap_info = load_umap_model(str(rna_umap_path))    
            else:    
                print(f"  训练{input_domain.upper()} UMAP模型...")    
                _, _ = train_and_save_umap_model(    
                    data=rna_latent,    
                    save_path=str(rna_umap_path),    
                    n_neighbors=umap_n_neighbors,    
                    min_dist=umap_min_dist,    
                    random_state=umap_random_state    
                )    
                input_umap_info = load_umap_model(str(rna_umap_path))    
            input_umap_model = input_umap_info['model']    
        else:  # protein    

            if protein_umap_path.exists():    
                input_umap_info = load_umap_model(str(protein_umap_path))    
            else:    
                print(f"  训练{input_domain.upper()} UMAP模型...")    
                _, _ = train_and_save_umap_model(    
                    data=protein_latent,    
                    save_path=str(protein_umap_path),    
                    n_neighbors=umap_n_neighbors,    
                    min_dist=umap_min_dist,    
                    random_state=umap_random_state    
                )    
                input_umap_info = load_umap_model(str(protein_umap_path))    
            input_umap_model = input_umap_info['model']    

        # 目标域UMAP模型（用于转换后数据的投影）    
        if target_domain == 'rna':   
            print("target_domain",target_domain)
            if rna_umap_path.exists():    
                target_umap_info = load_umap_model(str(rna_umap_path))    
            else:    
                print(f"  训练{target_domain.upper()} UMAP模型...")    
                _, _ = train_and_save_umap_model(    
                    data=rna_latent,    
                    save_path=str(rna_umap_path),    
                    n_neighbors=umap_n_neighbors,    
                    min_dist=umap_min_dist,    
                    random_state=umap_random_state    
                )    
                target_umap_info = load_umap_model(str(rna_umap_path))    
            target_umap_model = target_umap_info['model']    
        else:  # protein    
            if protein_umap_path.exists():    
                target_umap_info = load_umap_model(str(protein_umap_path))    
            else:    
                print(f"  训练{target_domain.upper()} UMAP模型...")    
                _, _ = train_and_save_umap_model(    
                    data=protein_latent,    
                    save_path=str(protein_umap_path),    
                    n_neighbors=umap_n_neighbors,    
                    min_dist=umap_min_dist,    
                    random_state=umap_random_state    
                )    
                target_umap_info = load_umap_model(str(protein_umap_path))    
            target_umap_model = target_umap_info['model']    
    elif dim_reduction == 'umap_independent':
        print("  降维方式设置为'umap_independent'，将对原始数据和转换数据分别独立进行UMAP降维")
    elif dim_reduction == 'none':    
        print("  降维方式设置为'none'，将直接使用数据的前两个维度进行绘图")    
    else:
        raise ValueError(f"不支持的降维方式: {dim_reduction}。支持的方式: 'none', 'umap', 'umap_independent'")
    
    # 加载模型    
    print("\n加载转换模型...")    
    translator = LatentTranslator(model_path, device=device)    
    translator.load_model(    
        input_dim1=rna_latent.shape[1],    
        input_dim2=protein_latent.shape[1],    
        hidden_dim=hidden_dim    
    )    
    
    # 选择输入数据    
    if input_domain == 'rna':    
        input_data = rna_latent    
    else:    
        input_data = protein_latent    
    if target_domain == 'rna':    
        target_original_data = rna_latent    
    else:    
        target_original_data = protein_latent    
    # 执行转换    
    print(f"\n执行转换: {mode_name}...")    
    if translate_method == "encode":
        output_data = translator.translate_encode(input_data, mode_tuple)    
    else:
        output_data = translator.translate(input_data, mode_tuple)    
    print(vars(translator.model))
    print(f"  输入形状: {input_data.shape}")    
    print(f"  输出形状: {output_data.shape}")  
    print(f"  输出原始形状: {target_original_data.shape}")  

    print("input_data.min()",input_data.min(),"input_data.max()",input_data.max())
    print("output_data.min()",output_data.min(),"output_data.max()",output_data.max())
    print("target_original_data.min()",target_original_data.min(),"target_original_data.max()",target_original_data.max())

    if model_state_path is not None:
        model_state_path = Path(model_state_path)
        model_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(translator.model.state_dict(), model_state_path)


    if translate_method != "encode":
        metrics_mode = calculate_metrics(target_original_data, output_data, mode)
        if mode_tuple in [(1, 2), (2, 1)]  :
            mode_inverse = (mode_tuple[1], mode_tuple[0])
            input_data_inverse = translator.translate(output_data, mode_inverse)
            metrics_mode_inverse = calculate_metrics(input_data, input_data_inverse, f"{mode_tuple}_{mode_inverse}_inverse")

        
    # 保存转换结果    
    print("\n保存转换结果...")    
    output_adata = sc.AnnData(output_data)    
    # 复制输入域的obs信息    
    if input_domain == 'rna':    
        output_adata.obs = rna_adata.obs.copy()    
    else:    
        output_adata.obs = protein_adata.obs.copy()    
    
    # 根据降维方式添加对应的嵌入
    if dim_reduction == 'umap':
        output_adata.obsm[f'X_umap_input_{input_domain}'] = input_umap_model.transform(input_data)
        output_adata.obsm[f'X_umap_output_{target_domain}'] = target_umap_model.transform(output_data)
    elif dim_reduction == 'umap_independent':
        # 对原始数据独立进行UMAP降维
        input_umap_model = None
        target_umap_model = None

        adata_orig_temp = sc.AnnData(input_data)
        sc.pp.neighbors(adata_orig_temp, n_neighbors=umap_n_neighbors, random_state=umap_random_state)
        sc.tl.umap(adata_orig_temp, min_dist=umap_min_dist, random_state=umap_random_state)
        output_adata.obsm[f'X_umap_independent_original'] = adata_orig_temp.obsm['X_umap']
        
        # 对转换后数据独立进行UMAP降维
        adata_trans_temp = sc.AnnData(output_data)
        sc.pp.neighbors(adata_trans_temp, n_neighbors=umap_n_neighbors, random_state=umap_random_state)
        sc.tl.umap(adata_trans_temp, min_dist=umap_min_dist, random_state=umap_random_state)
        output_adata.obsm[f'X_umap_independent_translated'] = adata_trans_temp.obsm['X_umap']
    else:
        input_umap_model = None
        target_umap_model = None
    if latent_key is None:    
        output_adata.obsm["X_latent"] = output_data    
    else:    
        output_adata.obsm[latent_key] = output_data    
    
    output_path = outdir / f'{mode}_output.h5ad'    
    output_adata.write_h5ad(output_path)    
    print(f"✓ 转换结果已保存: {output_path}")    
    
    # 绘制对比图    
    print("\n生成对比可视化...")    
    plot_path = outdir / f'{mode}_comparison.pdf'


    plot_comparison(    
        original_data=input_data,    
        translated_data=output_data,    
        labels=labels,    
        label_name=cell_type_key,
        mode_str=mode_name,    
        save_path=str(plot_path),    
        figsize=(12, 5),    
        dim_reduction=dim_reduction,    
        input_umap_model=input_umap_model,
        target_umap_model=target_umap_model,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state
    )    
    
    return {    
        'mode': mode,    
        'mode_name': mode_name,    
        'input_domain': input_domain,    
        'target_domain': target_domain,    
        'input_data': input_data,    
        'output_data': output_data,    
        'output_path': str(output_path),    
        'plot_path': str(plot_path),    
        'adata': output_adata,    
        'input_umap_path': str(rna_umap_path) if input_domain == 'rna' else str(protein_umap_path),    
        'target_umap_path': str(rna_umap_path) if target_domain == 'rna' else str(protein_umap_path)    
    }  


def translate_all_modes(  
    model_path: str,  
    rna_adata_path: str,  
    protein_adata_path: str,  
    latent_key: str = 'X_latent',  
    hidden_dim: int = 16,  
    outdir: str = './inference_output',  
    cell_type_key: str = 'cell_type',  
    device: str = 'cpu',  
    dim_reduction: str = "umap",  
    translate_method: str = 'decode',
    umap_n_neighbors: int = 15,  
    umap_min_dist: float = 0.5,  
    umap_random_state: int = 42  
) -> Dict[str, Optional[Dict[str, Any]]]:  
    """  
    执行所有四种转换模式  
      
    参数:  
        model_path: 模型路径  
        rna_adata_path: RNA 数据路径  
        protein_adata_path: Protein 数据路径  
        latent_key: 潜在表示的键名  
        hidden_dim: 隐藏层维度  
        outdir: 输出目录  
        cell_type_key: 细胞类型标签键  
        device: 计算设备  
        dim_reduction: 降维方式 ('none', 'umap', 'umap_independent')
        umap_n_neighbors: UMAP邻居数量  
        umap_min_dist: UMAP最小距离  
        umap_random_state: UMAP随机种子  
      
    返回:  
        包含所有转换结果的字典  
    """  
    modes = ['2to1', '1to2', '1to1', '2to2']  
    results = {}  

    # 确保输出根目录存在  
    root_outdir = Path(outdir)  
    root_outdir.mkdir(parents=True, exist_ok=True)  
      
    for mode in modes:  
        try:  
            mode_outdir = root_outdir / mode  
            result = translate_single_mode(  
                model_path=model_path,  
                rna_adata_path=rna_adata_path,  
                protein_adata_path=protein_adata_path,  
                mode=mode,  
                latent_key=latent_key,  
                hidden_dim=hidden_dim,  
                outdir=str(mode_outdir),  
                cell_type_key=cell_type_key,  
                device=device,  
                dim_reduction=dim_reduction,  
                translate_method=translate_method,
                umap_n_neighbors=umap_n_neighbors,  
                umap_min_dist=umap_min_dist,  
                umap_random_state=umap_random_state  
            )  
            results[mode] = result  
            print(f"✓ 模式 {mode} 完成\n")  
        except Exception as e:  
            print(f"✗ 模式 {mode} 失败: {e}\n")  
            import traceback  
            traceback.print_exc()  
            results[mode] = None  
      
    return results
