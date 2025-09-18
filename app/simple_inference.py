"""
simple_inference.py

This script performs token classification (e.g., NER) inference on a set of text files using a HuggingFace Transformers model.
It reads .txt files, runs the model, and writes .ann annotation files in BRAT format.

Usage example:
  python simple_inference.py -i <input_txt_dir> -o <output_ann_dir> -m <model_path> [--overwrite] [--agg_strat <strategy>]

Author: Jan Rodríguez Miret
"""

import re
import argparse
import sys
import torch
from transformers import pipeline
from spacy.lang.es import Spanish


def parse_args():
    parser = argparse.ArgumentParser(description="Specify variables for the Model Inference")
    parser.add_argument('-i', "--txts_path", required=True, type=str, help="Input directory containing the .txt text files")
    parser.add_argument('-m', "--model_path", required=True, type=str, help="Path to the model")
    parser.add_argument("-o", "--anns_path", required=False, type=str, help="Output directory for .ann annotation files")
    parser.add_argument("-ow", "--overwrite", action='store_true', default=False, help="Overwrite current .ann files in the output directory with new generated ones")
    parser.add_argument("-agg", "--agg_strat", default="first", type=str, help="Aggregation strategy. One of ('simple', 'first', 'max', or 'average'")
    return parser

def get_added_spaces(sentence, sentence_pretokenized):
    """
    Given an original sentence and its pretokenized version (with added spaces),
    return a list of indices where spaces were added in the pretokenized string.
    This is used to align model predictions back to the original text offsets.
    """
    # 'i' contains the current character index of 'sentence'
    # 'j' contains the current character index of 'sentence_pretokenized' (which has added_spaces)
    i = j = 0
    added_spaces = []
    while j < len(sentence_pretokenized):
        if sentence[i] == sentence_pretokenized[j]:
            i += 1
            j += 1
        elif sentence[i] == sentence_pretokenized[j+1] and sentence_pretokenized[j] == ' ':
            added_spaces.append(j)
            j += 1
        else:
            raise AssertionError("This should never be called.")
    return added_spaces

def align_results(results_pre, added_spaces, start_sent_offset):
    """
    Adjusts the entity offsets in the model's output to match the original text,
    correcting for any added spaces during pretokenization.
    Returns a list of aligned entity dictionaries.
    """
    aligned_results = []
    for entity in results_pre:
        aligned_entity = entity.copy()
        num_added_spaces_before = len(list(filter(lambda offset: offset < entity['start'], added_spaces)))
        num_added_spaces_after = len(list(filter(lambda offset: offset < entity['end'], added_spaces)))
        added_spaces_between = list(filter(lambda offset: (offset > entity['start']) & (offset < entity['end']), added_spaces))
        aligned_entity['word'] = entity['word'].strip()
        aligned_entity['word'] = ''.join([char for i, char in enumerate(aligned_entity['word']) if i + aligned_entity['start'] not in added_spaces_between])
        aligned_entity['span'] = aligned_entity.pop('word')
        aligned_entity['ner_class'] = aligned_entity.pop('entity_group')
        aligned_entity['start'] = start_sent_offset + entity['start'] - num_added_spaces_before
        aligned_entity['end'] = start_sent_offset + entity['end'] - num_added_spaces_after
        aligned_results.append(aligned_entity)
    return aligned_results

def write_to_ann(ann_path, results):
    """
    Write the model's entity predictions to a .ann file in BRAT format.
    Each entity is written as a line: T<ID>\t<LABEL> <START> <END>\t<TEXT>
    """
    results_ann_str = "\n".join([f"T{tid+1}\t{result['entity_group']} {result['start']} {result['end']}\t{result['word']}" for tid, result in enumerate(results)])
    with open(ann_path, "w+") as file:
        file.write(results_ann_str)

def join_all_entities(results):
    num_texts = len(results[0])  # number of documents
    entities_all = []

    for text_idx in range(num_texts):
        entities_file = []
        for model_idx in range(len(results)):
            entities_file.extend(results[model_idx][text_idx])
        entities_file = sorted(entities_file, key=lambda x: (x['start'], -x['end']))
        entities_all.append(entities_file)
    return entities_all


def ner_inference(texts, nerl_models_config, agg_strat="first", combined=False):

    # Regex for pretokenization (splitting words and punctuation)
    PRETOKENIZATION_REGEX = re.compile(
        r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])')
    
    # Load spaCy Spanish pipeline for sentence splitting
    nlp = Spanish()
    nlp.add_pipe("sentencizer")
    device = 0 if torch.cuda.is_available() else -1
    results = []
    for nerl_model_config in nerl_models_config:
        ner_model_path = nerl_model_config["ner_model_path"]
        pipe = pipeline("token-classification", model=ner_model_path, aggregation_strategy=agg_strat, device=device) # "simple" allows for different tags in a word, otherwise "first", "average", or "max".
        results_model = []
        # Process each .txt file
        for i_txt, text in enumerate(texts):
            lines = text.splitlines() 

            results_model_file = []
            
            # Track the offset of the start of each line in the file
            line_start_offset = 0
            for line in lines:
                doc = nlp(line)
                sents = list(doc.sents)
                for sentence in sents:
                    # Pretokenize sentence for model compatibility
                    pretokens = [t for t in PRETOKENIZATION_REGEX.split(sentence.text) if t]
                    # Add space between two non-space pretokens (we cannot join by whitespace directly,
                    # because of double whitespaces, tabs).
                    # This is necessary because the model expects tokens to be separated by spaces.
                    # The loop checks each pair of consecutive pretokens, and if both are not whitespace,
                    # it inserts a space between them. After inserting, it updates the length and index
                    # to account for the new space. This ensures that the pretokenized sentence matches
                    # the expected input format for the model, and that the mapping between original and
                    # pretokenized text can be reconstructed for offset alignment.
                    i_pret = 1
                    len_pretokens = len(pretokens)
                    while i_pret < len_pretokens:
                        if (not pretokens[i_pret-1].isspace() and not pretokens[i_pret].isspace()):
                            pretokens.insert(i_pret, " ")
                            len_pretokens = len(pretokens)
                            i_pret += 1 # Move one more because we added one before
                        i_pret += 1
                    sentence_pretokenized = ''.join(pretokens)
                    # Find where spaces were added
                    added_spaces = get_added_spaces(sentence.text, sentence_pretokenized)
                    # Run model inference
                    results_pre = pipe(sentence_pretokenized)
                    # Convert numpy types to native Python types for JSON serialization
                    for entity in results_pre:
                        entity['score'] = round(float(entity['score']), 4)
                        entity['ner_score'] = entity.pop('score')
                    # Align model results to original text offsets
                    # Use sentence.start_char + line_start_offset for robust offset alignment
                    results_sent = align_results(results_pre, added_spaces, sentence.start_char + line_start_offset)
                    results_model_file.extend(results_sent)
                line_start_offset += len(line)
            results_model.append(results_model_file)
            print(f"Finished {i_txt+1}/{len(texts)} ({round((i_txt+1)*100/len(texts), 3)}%)")
        # NOTE: Here we could also do normalization and link all mentions from results_model
        results.append(results_model)
    if combined:
        return join_all_entities(results)
    return results


def main(argv):
    """
    This is just for testing purposes.
    """
    texts = [
        "Este es un texto de ejemplo.\nCon un paciente procedente de Almería aunque nacido en Guadalupe, México, con mucha tos, mocos, fiebre y la varicela con meningitis.",
        "Otro texto con Covid y paracetamol para probar.\nCon más  muchos más síntomas interesantes como edemas."
    ]
    model_paths = ["/home/jan/bsc/best-l2kx7y5e", "/home/jan/bsc/location-sub-tagger"]
    results = ner_inference(texts, model_paths, agg_strat="first")
    print(results)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
