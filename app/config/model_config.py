from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_path: str="dmis-lab/biobert-base-cased-v1.1"
    tokenizer_path: str="dmis-lab/biobert-base-cased-v1.1"
    classifier_layers: List[int] = None
    dropout: float=0.2
    output_dim: int=4
    max_length: int=128
    model_save_path: str="saved_models/MuMetastasisDetection/"
    def __init__(self):
        self.classifier_layers = []