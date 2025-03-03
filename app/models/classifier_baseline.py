from models.model_classification import ModelClassification

from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from typing import Dict
from accelerate import Accelerator

class BinaryBERT(nn.Module):
    def __init__(
            self,
            config: Dict
        ):

        super(BinaryBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_path)
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
    
class PredictionPipeline(ModelClassification):
    def __init__(
            self,
            model: nn.Module,
            config: Dict,
            accelerator: Accelerator
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        if accelerator:
            self.model, self.device = accelerator.prepare(model)
        self.model.eval()

    def predict_text(
            self,
            text: str,
            footer: Dict
        ) -> int:

        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.config.max_length, truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
        return self.serialize(text, outputs[0].tolist(), footer)
        
"""

class ClassifierModel(ModelAnnotation):
    def __init__(self, csv_path):
        import csv
        import spacy
        from spacy.tokens import Doc, Span
        from spacy.language import Language

        self.nlp = spacy.load("es_core_news_sm")
        self.entities = self.load_entities_from_csv(csv_path)

        @Language.component("dictionary_entity_recognizer")
        def dictionary_entity_recognizer(doc):
            matches = []
            for token in doc:
                if token.lower_ in self.entities:
                    start = token.i
                    end = token.i + 1
                    entity_info = self.entities[token.lower_]
                    matches.append(Span(doc, start, end, label=entity_info["label"]))

            new_ents = []
            for span in matches:
                overlap = False
                for ent in doc.ents:
                    if span.start < ent.end and span.end > ent.start:
                        overlap = True
                        break
                if not overlap:
                    new_ents.append(span)

            doc.ents = list(doc.ents) + new_ents
            return doc

        self.nlp.add_pipe("dictionary_entity_recognizer", last=True)

    def load_entities_from_csv(self, file_path):
        entities = {}
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 5:  # Ensure there are enough columns
                    entities[row[0].lower()] = {
                        "label": row[1].lower(),
                        "dt4h_concept_identifier": row[2],
                        "nel_component_type": row[3],
                        "nel_component_version": row[4]
                    }
        return entities

    def predict(self, text, app, footer):
        doc = self.nlp(text)
        annotations = []

        for ent in doc.ents:
            entity_text = ent.text.lower()
            if entity_text in self.entities:
                entity_info = self.entities[entity_text]
                annotation = {
                    "concept_class": ent.label_,
                    "start_offset": ent.start_char,
                    "end_offset": ent.end_char,
                    "concept_mention_string": ent.text,
                    "concept_confidence": 0.95,  # Example value
                    "ner_component_type": "dictionary lookup",
                    "ner_component_version": self.nlp.meta["version"],
                    "negation": "no",  # Simplified
                    "negation_confidence": 1.0,
                    "qualifier_negation": "",
                    "qualifier_temporal": "",
                    "dt4h_concept_identifier": entity_info["dt4h_concept_identifier"],
                    "nel_component_type": entity_info["nel_component_type"],
                    "nel_component_version": entity_info["nel_component_version"],
                    "controlled_vocabulary_namespace": "none",
                    "controlled_vocabulary_version": "",
                    "controlled_vocabulary_concept_identifier": "",
                    "controlled_vocabulary_concept_official_term": "",
                    "controlled_vocabulary_source": "original"
                }
                annotations.append(annotation)
            else:
                print(f"Entity '{entity_text}' not found in dictionary.")

        return self.serialize(text, annotations, footer)
"""