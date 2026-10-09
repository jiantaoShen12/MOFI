import yaml
from typing import Union, Dict, Any, Optional
from pathlib import Path

def load_config(
    config: Union[str, Path, Dict[str, Any]], 
    *, 
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Loads a configuration from a name, a file path, or a dictionary.
    Enhanced version: supports relative paths from project_root.

    Parameters
    ----------
    config
        - If a dictionary, returned directly.
        - If a Path or absolute path string, loaded as YAML.
        - If a relative path string (contains '/'), resolved from project_root.
        - If a simple name string, loaded from built-in configs/.
    project_root
        Base directory for resolving relative paths (e.g., config_253582/xxx.yaml).
        If None, only checks absolute paths and built-in configs.

    Returns
    -------
    Dict[str, Any]
        The loaded configuration dictionary.
    """
    # Case 1: Direct dictionary
    if isinstance(config, dict):
        return config

    if not isinstance(config, (str, Path)):
        raise TypeError(f"config must be str, Path, or dict, got {type(config)}")

    config_str = str(config)
    config_path = Path(config_str)

    # Case 2: Absolute path or file exists as-is
    if config_path.is_absolute() or config_path.is_file():
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading custom config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # Case 3 替代方案：自动向上查找
    if '/' in config_str:
        # 从当前工作目录或脚本位置开始查找
        search_dirs = [
            Path.cwd(),                                    # 当前工作目录
            Path(__file__).parent.parent.parent,          # 项目根目录（CytoBridge/ 的父级）
        ]
        for base_dir in search_dirs:
            candidate = base_dir / config_path
            if candidate.is_file():
                print(f"Loading config from: {candidate}")
                with open(candidate, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
                    
    # Case 4: Simple name -> built-in config (e.g., "unbalanced_ot_253")
    config_name = config_str.lower().replace('.yaml', '').replace('.yml', '')
    
    try:
        # Built-in configs directory: cytobridge/configs/
        pkg_config_dir = Path(__file__).parent.parent / "configs"
        
        built_in_path = pkg_config_dir / f"{config_name}.yaml"

        if built_in_path.is_file():
            print(f"Loading built-in config: '{config_name}'")
            with open(built_in_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # List available built-in configs
            available = sorted([
                f.stem for f in pkg_config_dir.glob('*.yaml') 
                if not f.stem.startswith('_')
            ])
            raise FileNotFoundError(
                f"Config '{config}' not found. Tried:\n"
                f"  - Absolute path: {config_path.absolute()}\n"
                f"  - Project root ({project_root}): {project_root / config_path if project_root else 'N/A'}\n"
                f"  - Built-in configs: {available}"
            )
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise RuntimeError(
            f"Error loading config '{config}': {e}"
        )