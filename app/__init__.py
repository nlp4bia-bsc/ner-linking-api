from flask import Flask, request, jsonify
#from flasgger import Swagger
from simple_inference import ner_inference
from config import NERL_MODELS_CONFIG

app = Flask(__name__)
#swagger = Swagger(app)



@app.route('/nerl_process_bulk', methods=['POST'])
def nerl_process_bulk():
    """
    Process multiple texts using the NER models and normalizing to the provided gazetteers.
    ---
    parameters:
      - content: list
        - text: str
      - nerl_models_config: list
        - ner_model_path: str
        - gazetteer_path: str
    responses:
      200:
        description: Processed texts with NER annotations normalized.
    """
    body = request.json

    if not isinstance(body, dict) or not "content" in body.keys():
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    content = body["content"]
    nerl_models_config = body.get("nerl_models_config", NERL_MODELS_CONFIG)

    if not isinstance(content, list):
        return jsonify({"error": "Input must be a list of objects"}), 400
    
    texts = []
    for item in content:
            text = item.get('text')
            if not text:
                return jsonify({"error": "Each item must contain 'text'"}), 400
            texts.append(text)

    results = ner_inference(texts, nerl_models_config, agg_strat="first")

    # TODO: Here we will add normalization to the gazetteers

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
