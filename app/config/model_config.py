from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

base_path = Path(__file__).parent.parent # app/

"""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dmis-lab/biobert-base-cased-v1.1", 
    cache_dir="./huggingface_mirror"
)

to download any model from the Hugging Face Hub in a specific directory.
"""
@dataclass
class ModelConfig:
    model_name: str="dmis-lab/biobert-base-cased-v1.1"
    cache_dir: str=str(base_path / "models" / "huggingface_mirror")
    classifier_layers: List[int] = None
    dropout: float=0.2
    output_dim: int=4
    max_length: int=128
    model_save_path: str=str(base_path / "saved_models" / "MuMetastasisDetection" / "model.pth")
    def __init__(self):
        self.classifier_layers = []