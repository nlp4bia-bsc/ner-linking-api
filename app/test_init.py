import argparse

from config.model_config import ModelConfig
from models.classifier_baseline import BinaryBERT, PredictionPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the script.")

    parser.add_argument(
        "--text", 
        type=str, 
        default="",
        help="Text to process"
    )
    return parser.parse_args()

def main(text):
    config = ModelConfig()
    config.__init__()
    model = BinaryBERT(config)
    pipeline = PredictionPipeline(model, config)
    random_footer = {
        "provider_id": "1",
        "person_id": "2",
        "visit_detail_id": "3",
        "note_id": "4",
        "note_type_concept_id": "5",
        "note_datetime": "6",
        "note_title": "7"
    }
    return pipeline.predict(text, random_footer)

if __name__ == "__main__":
    args = parse_arguments()
    text = args.text
    print(main(text))