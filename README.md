# Named Entity Recognition + Linking (NERL) Inference API

This is a Flask API that performs named entity recognition (NER) and linking/normalization (NEL) using any Hugging Face token classification model of your preference for NER and configuring the gazetters, and sentence similarity & text-reranking models for normalization.

## 💻 Prerequisites

⚠ This API will work much faster if it has access to a GPU, though it can also work with CPU only at a much slower pace.

First, create a virtual environment, activate it, and install `requirements.txt`.
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
```

### 🛠 (Recommended) Configure the default models and gazetteers

To avoid sending the `nerl_models_config` on each request, you can just configure the default models and gazetteers to be used in `app/config.py`. Note that model paths can also be Hugging Face identifiers (e.g. *BSC-NLP4BIA/bsc-bio-ehr-es-carmen-symptemist*). In such case, the models will be downloaded the first time and saved in local cache for future uses.

Anyway, if you provide them in the HTTP request, they will override the default ones specified.

## 🏃‍♀️ Running the service

To start the service, just run the `__init__.py` file.
```
python3 __init__.py
```

This will create a Flask API web server listening on port 5000 (or the one you defined). Take into account that it will be on *development mode*, meaning that it cannot handle multiple simultaneous requests correctly, it does not auto-restart, and we don't have TLS certificates so that it can be called using HTTPS, only HTTP.

We should get something like:
```
user@computer$ python __init__.py 
 * Serving Flask app '__init__'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: XXX-YYY-ZZZ

```

## Available endpoints

These endpoints can be called from Python using its built-in `requests` package, any other HTTP client library from any programming language, `curl`, Postman, etc.

- `nerl_process_bulk`: perform named entity recognition and linking using the specified models and gazetteers (or default ones) to all texts provided. It works for multiple models, which will be run sequentially and their outputs combined, with mentions ordered by their order of appearance (offset) in text.
```
parameters:
      - content: list
        - text: str
      - nerl_models_config: list
        - ner_model_path: str (local path or HF identifier)
        - nel_model_path: str (local path or HF identifier)
        - gazetteer_path: str
        - vectorized_gazetteer_path: str (optional)
```
Example:
- Input
```
{
  "content": [
    {
      "text": "Paciente con mucha tos, mocos y fiebre. Test de Gripe A negativo."
    },
    {
      "text": "Mujer diabética acude a urgencias desorientada y con convulsiones."
    }
  ]
}
```
- Output
```
[
    [
        {
            "code": "49727002",
            "end": 22,
            "nel_score": 1.0,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "tos",
            "start": 19,
            "term": "tos"
        },
        {
            "code": "301291007",
            "end": 29,
            "nel_score": 0.7996,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "mocos",
            "start": 24,
            "term": "esputo acuoso"
        },
        {
            "code": "64882008",
            "end": 38,
            "nel_score": 1.0,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "fiebre",
            "start": 32,
            "term": "fiebre"
        },
        {
            "code": "441119003",
            "end": 64,
            "nel_score": 0.831,
            "ner_class": "SINTOMA",
            "ner_score": 0.9997,
            "span": "Test de Gripe A negativo",
            "start": 40,
            "term": "prueba para la detección de virus respiratorios (adenovirus, rinovirus, virus sincitial respiratorio, parainfluenza e influenza) negativa"
        },
        {
            "code": "442438000",
            "end": 55,
            "nel_score": 0.9106,
            "ner_class": "ENFERMEDAD",
            "ner_score": 0.9997,
            "span": "Gripe A",
            "start": 48,
            "term": "gripe causada por virus Influenza A"
        }
    ],
    [
        {
            "code": "73211009",
            "end": 15,
            "nel_score": 0.7994,
            "ner_class": "ENFERMEDAD",
            "ner_score": 0.9939,
            "span": "diabética",
            "start": 6,
            "term": "diabetes mellitus"
        },
        {
            "code": "62476001",
            "end": 46,
            "nel_score": 0.9681,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "desorientada",
            "start": 34,
            "term": "desorientado"
        },
        {
            "code": "91175000",
            "end": 65,
            "nel_score": 0.9399,
            "ner_class": "SINTOMA",
            "ner_score": 0.9999,
            "span": "convulsiones",
            "start": 53,
            "term": "convulsión"
        }
    ]
]
```

## Docker Support (in progress)