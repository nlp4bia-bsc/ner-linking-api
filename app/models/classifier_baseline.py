from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from typing import Dict

from app.models.model_classification import ModelClassification

class BinaryBERT(nn.Module):
    def __init__(
            self,
            config: Dict
        ):

        super(BinaryBERT, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            local_files_only=True
        )
        last_size = self.bert.config.hidden_size
        layers = []
        for num_dims in config.classifier_layers:
            layers.append(nn.Dropout(config.model.dropout))
            layers.append(nn.Linear(last_size, num_dims))
            layers.append(nn.ReLU())
            last_size = num_dims
        layers.append(nn.Linear(last_size, config.output_dim))
        layers.append(nn.Softmax(dim=-1))
        self.classifier_layers = nn.Sequential(*layers)

        self._load_model()

    def forward(
            self, 
            input_ids: torch.Tensor, 
            attention_mask: torch.Tensor
        ) -> torch.Tensor: # no input ids since we are not passing more than one input

        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
            )
        
        hidden_states = outputs['last_hidden_state']
        # (bs, seq_len, hidden_size)
        pooled_output = hidden_states[:, 0] # leverages the design of the [CLS] token as a compact representation of the entire input
        # (bs, hidden_size)
        return self.classifier_layers(pooled_output) # (bs, output_dim)
    
    def _load_model(self):
        self.load_state_dict(torch.load(self.config.model_save_path, map_location=torch.device('cpu'))) # since we are using cpu only
    
class PredictionPipeline(ModelClassification):
    def __init__(
            self,
            model: nn.Module,
            config: Dict,
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            local_files_only=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.model.eval()

    def predict(
            self,
            text: str,
            footer: Dict
        ) -> int:

        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.config.max_length, truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
        return self.serialize(text, outputs[0].tolist(), footer)