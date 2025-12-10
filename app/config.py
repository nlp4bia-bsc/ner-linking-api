NERL_MODELS_CONFIG = [
    {
        "ner_model_path": "BSC-NLP4BIA/bsc-bio-ehr-es-carmen-distemist",
        "nel_model_path": "ICB-UMA/ClinLinker-KB-GP",
        "gazetteer_path": "gazetteers/dictionary_distemist.tsv",
        "vectorized_gazetteer_path": "gazetteers/vectorized_distemist_gazetteer_es.pt",
     },
    {
        "ner_model_path": "BSC-NLP4BIA/bsc-bio-ehr-es-carmen-symptemist",
        "nel_model_path": "ICB-UMA/ClinLinker-KB-GP",
        "gazetteer_path": "gazetteers/symptemist_gazetter_snomed_ES_v2.tsv",
        "vectorized_gazetteer_path": "gazetteers/vectorized_symptemist_gazetteer_es.pt",
     },
]

# Negation/Uncertainty tagger model path
NEGATION_TAGGER_MODEL_PATH = "BSC-NLP4BIA/negation-tagger"