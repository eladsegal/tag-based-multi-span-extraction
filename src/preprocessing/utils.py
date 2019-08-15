import os
from typing import Dict, List, Tuple

from allennlp.data.tokenizers import Token


TRAIN_PATH = os.path.join('data', 'drop_dataset_train.json')

SPAN_ANSWER_TYPE = 'span'
NUMBER_ANSWER_TYPE = 'number'
DATE_ANSWER_TYPE = 'date'
SINGLE_SPAN = 'single_span'
MULTIPLE_SPAN = 'multiple_span'
SPAN_ANSWER_TYPES = [SINGLE_SPAN, MULTIPLE_SPAN]
ANSWER_TYPES = SPAN_ANSWER_TYPES + ['number', 'date']


def get_answer_type(answer):
    if answer['number']:
        return 'number'
    elif answer['spans']:
        if len(answer['spans']) == 1:
            return SINGLE_SPAN
        return MULTIPLE_SPAN
    elif any(answer['date'].values()):
        return 'date'
    else:
        return None


def find_span(answer_tokens: List[Token], qp_token_indices: Dict[Token, List[int]],
              num_qp_tokens) -> List[Tuple[int, int]]:
    num_answer_tokens = len(answer_tokens)
    span = []
    for span_start in qp_token_indices[answer_tokens[0]]:
        if num_answer_tokens == 1:
            span.append((span_start, span_start))
        elif span_start + num_answer_tokens - 1 <= num_qp_tokens:
            for i, answer_token in enumerate(answer_tokens[1:], 1):
                if not span_start + i in qp_token_indices[answer_token]:
                    break
            else:
                span_end = span_start + i  # span_end is inclusive
                span.append((span_start, span_end))
    return span


def deep_dict_update(d, u):
    # based on https://stackoverflow.com/a/3233356/2133678
    for k, v in u.items():
        dv = d.get(k, {})
        if not isinstance(dv, dict):
            d[k] = v
        elif isinstance(v, dict):
            d[k] = deep_dict_update(dv, v)
        else:
            d[k] = v
    return d

