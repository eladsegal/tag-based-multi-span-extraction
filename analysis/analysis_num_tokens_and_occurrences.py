import torch

import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse
import os
import pickle
import json
import types
from collections import defaultdict

from tqdm import tqdm

from allennlp.data.instance import Instance
from allennlp.data.fields import (Field, TextField, IndexField, ListField,
                                  MetadataField, ArrayField, SequenceLabelField, SpanField)
from allennlp.data.tokenizers import Token
from allennlp.common.util import import_submodules
from allennlp.tools.drop_eval import _answer_to_bags, _compute_f1, _match_numbers_if_present

from src.data.tokenizers.huggingface_transformers_tokenizer import HuggingfaceTransformersTokenizer
from src.data.dataset_readers.utils import (custom_word_tokenizer, split_tokens_by_hyphen, 
                                            index_text_to_tokens, find_valid_spans)
from src.data.dataset_readers.drop.drop_utils import extract_answer_info_from_annotation

def align_predicted_and_maximizing_gold(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    alignement = _align_bags(predicted_bags[1], gold_bags[1])
    return alignement

def _align_bags(predicted, gold):
    """
    Takes gold and predicted answer sets and finds the optimal 1-1 alignment
    between them.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    alignment = [-1] * len(gold)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        prev_max_score = max_scores[row]
        max_scores[row] = max(max_scores[row], scores[row, column])
        if max_scores[row] > prev_max_score:
            alignment[row] = column
    return alignment

MAX_PIECES = 512

def get_analysis(dataset_pkl_path, predictions_path, heads):
    with open(os.path.join(dataset_pkl_path), 'rb') as dataset_pkl:
        dataset = pickle.load(dataset_pkl)

    predictions = {}
    with open(os.path.join(predictions_path), 'rb') as predictions_file:
        while True:
            line = predictions_file.readline()
            if not line:
                break
            prediction = json.loads(line)
            predictions[prediction['query_id']] = prediction

    tokenizer = HuggingfaceTransformersTokenizer(pretrained_model='roberta-large')
    word_tokenizer = custom_word_tokenizer()
    word_tokenize =\
        lambda text: [token for token in split_tokens_by_hyphen(word_tokenizer.tokenize(text))]


    num_gold_tokens_to_stats = defaultdict(lambda: defaultdict(int))
    num_occurrences_stats = defaultdict(lambda: defaultdict(int))
    for i, instance in tqdm(enumerate(dataset)):
        full_prediction = predictions[instance['metadata']['question_id']]
        if full_prediction['predicted_ability'] not in heads:
            continue

        question_text = instance['metadata']['original_question']
        question_tokens = tokenizer.tokenize_with_offsets(question_text)
        question_text_index_to_token_index = index_text_to_tokens(question_text, question_tokens)
        question_words = word_tokenize(question_text)
        question_alignment = tokenizer.align_tokens_to_tokens(question_text, question_words, question_tokens)
        question_wordpieces = tokenizer.alignment_to_token_wordpieces(question_alignment)

        passage_text = instance['metadata']['original_passage']
        passage_tokens = tokenizer.tokenize_with_offsets(passage_text)
        passage_text_index_to_token_index = index_text_to_tokens(passage_text, passage_tokens)
        passage_words = word_tokenize(passage_text)
        passage_alignment = tokenizer.align_tokens_to_tokens(passage_text, passage_words, passage_tokens)
        passage_wordpieces = tokenizer.alignment_to_token_wordpieces(passage_alignment)
        passage_text = passage_text[:instance['metadata']['max_passage_length']]

        question_passage_tokens = instance['metadata']['question_passage_tokens']
        question_passage_wordpieces = instance['metadata']['question_passage_wordpieces']

        # Index tokens
        encoded_inputs = tokenizer.encode_plus([token.text for token in question_tokens], [token.text for token in passage_tokens], 
                                    add_special_tokens=True, max_length=MAX_PIECES, 
                                    truncation_strategy='only_second',
                                    return_token_type_ids=True,
                                    return_special_tokens_mask=True)
        question_passage_token_type_ids = encoded_inputs['token_type_ids']
        question_passage_special_tokens_mask = encoded_inputs['special_tokens_mask']

        question_position = tokenizer.get_type_position_in_sequence(0, question_passage_token_type_ids, question_passage_special_tokens_mask)
        passage_position = tokenizer.get_type_position_in_sequence(1, question_passage_token_type_ids, question_passage_special_tokens_mask)
        question_passage_tokens, num_of_tokens_per_type = tokenizer.convert_to_tokens(encoded_inputs, [
            {'tokens': question_tokens, 'wordpieces': question_wordpieces, 'position': question_position}, 
            {'tokens': passage_tokens, 'wordpieces': passage_wordpieces, 'position': passage_position}
        ])

        # Adjust text index to token index
        question_text_index_to_token_index = [token_index + question_position for i, token_index in enumerate(question_text_index_to_token_index)
                                              if token_index < num_of_tokens_per_type[0]]
        passage_text_index_to_token_index = [token_index + passage_position for i, token_index in enumerate(passage_text_index_to_token_index)
                                              if token_index < num_of_tokens_per_type[1]]

        prediction = full_prediction['answer']['value']
        maximizing_ground_truth = full_prediction['maximizing_ground_truth']
        answer_accessor, answer_texts = extract_answer_info_from_annotation(maximizing_ground_truth)

        gold_indexes = {'question': None, 'passage': None}

        alignment = align_predicted_and_maximizing_gold(prediction, answer_texts)
        for gold_index, predicted_index in enumerate(alignment):
            num_of_gold_tokens = len(tokenizer.tokenize(answer_texts[gold_index]))
            num_of_predicted_tokens = len(tokenizer.tokenize(prediction[predicted_index]))
            num_gold_tokens_to_stats[num_of_gold_tokens]['count'] += 1
            num_gold_tokens_to_stats[num_of_gold_tokens]['num_predicted_tokens'] += num_of_predicted_tokens
            num_gold_tokens_to_stats[num_of_gold_tokens]['em'] += full_prediction['em'] * 100
            num_gold_tokens_to_stats[num_of_gold_tokens]['f1'] += full_prediction['f1'] * 100

            answer_spans = []
            answer_spans += find_valid_spans(question_text, [answer_texts[gold_index]], 
                                            question_text_index_to_token_index, 
                                            question_passage_tokens, question_passage_wordpieces, 
                                            gold_indexes['question'])
            answer_spans += find_valid_spans(passage_text, [answer_texts[gold_index]], 
                                                    passage_text_index_to_token_index, 
                                                    question_passage_tokens, question_passage_wordpieces, 
                                                    gold_indexes['passage'])
            occurrences = len(answer_spans)
            num_occurrences_stats[occurrences]['count'] += 1
            num_occurrences_stats[occurrences]['em'] += full_prediction['em'] * 100
            num_occurrences_stats[occurrences]['f1'] += full_prediction['f1'] * 100

    return num_gold_tokens_to_stats, num_occurrences_stats
