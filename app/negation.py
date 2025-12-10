"""
negation.py

This module handles negation and uncertainty detection on NER results.
It uses a negation/uncertainty tagger model to identify NEG/NSCO/UNC/USCO entities
and enriches NER entities with is_negated and is_uncertain attributes.

Author: Jan Rodríguez Miret
"""

from simple_inference import _process_texts_with_model
from config import NEGATION_TAGGER_MODEL_PATH


def negation_tagger_inference(texts, negation_tagger_model_path, agg_strat="first"):
    """
    Run negation/uncertainty tagger on texts to detect negation and uncertainty scopes.
    
    Args:
        texts (list): List of text strings to process
        negation_tagger_model_path (str): Path or HF identifier for the negation tagger model
        agg_strat (str): Aggregation strategy for token classification ('simple', 'first', 'max', 'average')
    
    Returns:
        list: List of lists containing negation/uncertainty entities for each text
              Each entity is a dict with: span, start, end, ner_class, ner_score
              Entity types: NEG, NSCO (negation scope), UNC, USCO (uncertainty scope)
    """
    return _process_texts_with_model(texts, negation_tagger_model_path, agg_strat=agg_strat)


def add_negation_uncertainty_attributes(nerl_entities_list, texts, agg_strat="first"):
    """
    Add is_negated and is_uncertain attributes to NER entities based on overlap with negation/uncertainty scope entities.
    Also includes negation/uncertainty scores from the matching scopes.
    Calls the negation tagger internally to detect negation/uncertainty scopes.
    Filters out NEG, NSCO, UNC, USCO entities from the results.
    
    Args:
        ner_entities_list (list): List of NER entities for each text (output from ner_inference with combined=True)
        texts (list): List of original text strings to process with negation tagger
        agg_strat (str): Aggregation strategy for the negation tagger model
    
    Returns:
        list: List of NER entities with added is_negated, is_uncertain, negation_score, and uncertainty_score attributes
    """
    # Run negation tagger inference to get negation/uncertainty entities
    negation_entities_list = negation_tagger_inference(texts, NEGATION_TAGGER_MODEL_PATH, agg_strat=agg_strat)
    
    results_with_attributes = []
    
    for ner_entities_doc, negation_entities_doc in zip(nerl_entities_list, negation_entities_list):
        
        # Separate negation/uncertainty scopes and triggers
        neg_scopes = [e for e in negation_entities_doc if e['ner_class'] == 'NSCO']
        unc_scopes = [e for e in negation_entities_doc if e['ner_class'] == 'USCO']
        
        # Filter NER entities: exclude NEG, NSCO, UNC, USCO entities
        filtered_ner_entities = [
            e for e in ner_entities_doc 
            if e['ner_class'] not in ['NEG', 'NSCO', 'UNC', 'USCO']
        ]
        
        # Add negation/uncertainty attributes to each entity
        for entity in filtered_ner_entities:
            # Find overlapping negation scopes and get their scores
            overlapping_neg_scopes = [scope for scope in neg_scopes if _entity_in_scope(entity, scope)]
            is_negated = len(overlapping_neg_scopes) > 0
            # Use the highest score if multiple scopes overlap
            negation_score = max([scope['ner_score'] for scope in overlapping_neg_scopes]) if overlapping_neg_scopes else None
            
            # Find overlapping uncertainty scopes and get their scores
            overlapping_unc_scopes = [scope for scope in unc_scopes if _entity_in_scope(entity, scope)]
            is_uncertain = len(overlapping_unc_scopes) > 0
            # Use the highest score if multiple scopes overlap
            uncertainty_score = max([scope['ner_score'] for scope in overlapping_unc_scopes]) if overlapping_unc_scopes else None
            
            entity['is_negated'] = is_negated
            entity['is_uncertain'] = is_uncertain
            entity['negation_score'] = negation_score
            entity['uncertainty_score'] = uncertainty_score
        
        results_with_attributes.append(filtered_ner_entities)
    
    return results_with_attributes


def _entity_in_scope(entity, scope_ent):
    """
    Check if two entities overlap based on their start and end positions.
    Returns True if entity1 is within or overlaps with entity2's scope.
    
    Args:
        entity (dict): Entity (clinical) with 'start' and 'end' keys
        scope_ent (dict): Entity (NSCO/USCO) with 'start' and 'end' keys
    
    Returns:
        bool: True if entities overlap, False otherwise
    """
    return (
        # entity starts within scope
        (scope_ent['start'] <= entity['start'] < scope_ent['end']) or
        # entity ends within scope
        (scope_ent['start'] < entity['end'] <= scope_ent['end'])
        # TODO: Uncomment for considering negated the case where scope is "nested-smaller" in 
        # brat-peek, i.e., clin_ent completely contains NSCO/USCO ([cancer not detected])
        # or (entity1['start'] <= entity2['start'] and entity1['end'] >= entity2['end'])
    )
