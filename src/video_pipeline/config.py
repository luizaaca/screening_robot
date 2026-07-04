import yaml
from pathlib import Path
from typing import Optional
from video_pipeline.contracts import PipelineConfig

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"

def load_pipeline_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Carrega um arquivo YAML de configuracao e retorna um PipelineConfig validado.
    
    Se config_path for None, carrega `video_pipeline/configs/default.yaml`.
    """
    path_to_load = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if not path_to_load.exists():
        if config_path:
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path_to_load}")
        else:
            # Fallback seguro caso default.yaml seja deletado, 
            # as classes Pydantic proverão os defaults.
            return PipelineConfig()
            
    with open(path_to_load, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    if not data:
        return PipelineConfig()
        
    return PipelineConfig(**data)
