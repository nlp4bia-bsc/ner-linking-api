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

To avoid sending the `nerl_models_config` on each request, you can just configure the default models and gazetteers to be used in `app/config.py`.
Anyway, if you provide them in the request, they will override the default ones specified.

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
        - ner_model_path: str
        - gazetteer_path: str
```
Example:
- Input
```
{
    "content": [
        {
            "text": "Este es un texto de ejemplo con un paciente con mucha tos, mocos y fiebre, procedente de Almería, que además tenía edemas."
        },
        {
            "text": "Paciente femenina de 46 años que llega a urgencias con convulsiones graves."
        }
    ]
}
```
- Output
```
[
    [
        {
            "end": 57,
            "entity_group": "SINTOMA",
            "score": 0.9999017715454102,
            "start": 54,
            "word": "tos"
        },
        {
            "end": 64,
            "entity_group": "SINTOMA",
            "score": 0.999840497970581,
            "start": 59,
            "word": "mocos"
        },
        {
            "end": 73,
            "entity_group": "SINTOMA",
            "score": 0.9998804330825806,
            "start": 67,
            "word": "fiebre"
        },
        {
            "end": 96,
            "entity_group": "LUGAR_NATAL",
            "score": 0.9973543882369995,
            "start": 89,
            "word": "Almería"
        },
        {
            "end": 121,
            "entity_group": "SINTOMA",
            "score": 0.9940962791442871,
            "start": 115,
            "word": "edemas"
        }
    ],
    [
        {
            "end": 74,
            "entity_group": "SINTOMA",
            "score": 0.9996104836463928,
            "start": 55,
            "word": "convulsiones graves"
        }
    ]
]
```

## Docker Support (in progress)