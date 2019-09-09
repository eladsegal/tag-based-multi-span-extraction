import html
import re
import json
import os
from typing import Dict, List, Tuple


from allennlp.data.tokenizers import Token


TRAIN_PATH = os.path.join('data', 'drop_dataset_train.json')

SPAN_ANSWER_TYPE = 'spans'
NUMBER_ANSWER_TYPE = 'number'
DATE_ANSWER_TYPE = 'date'
SINGLE_SPAN = 'single_span'
MULTIPLE_SPAN = 'multiple_span'
SPAN_ANSWER_TYPES = [SINGLE_SPAN, MULTIPLE_SPAN]
ALL_ANSWER_TYPES = SPAN_ANSWER_TYPES + ['number', 'date']


def load_dataset(path):
    with open(path) as dataset_file:
        return json.load(dataset_file)


def save_dataset(dataset, path):
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=2)


def get_answer_type(answer):
    if answer['number']:
        return NUMBER_ANSWER_TYPE
    elif answer['spans']:
        if len(answer['spans']) == 1:
            return SINGLE_SPAN
        return MULTIPLE_SPAN
    elif any(answer['date'].values()):
        return DATE_ANSWER_TYPE
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

def fill_token_indices(tokens, text, uncased):
    new_tokens = []    
    text_idx = 0

    if uncased:
        text = text.lower()

    for token in tokens:
        first_char_idx = 2 if len(token.text) > 2 and token.text[:2] == "##" else 0

        while text[text_idx] == ' ' or text[text_idx] == '\xa0':
            text_idx += 1
        
        new_tokens.append(Token(text=token.text, idx = text_idx))             
        
        token_len = len(token.text) - first_char_idx

        if token.text == '[UNK]':
            token_len = 1

        text_idx += token_len

    return new_tokens

def token_to_span(token):
    start = token.idx
    end = token.idx + len(token.text)

    if token.text.startswith("##"):
        end -= 2

    if token.text == '[UNK]':
        end -= 4
    return (start, end)


def standardize_dataset(dataset):
    for passage_info in dataset.values():
        passage_info['passage'] = standardize_text(passage_info['passage'])
        for qa_pair in passage_info["qa_pairs"]:
            qa_pair['question'] = standardize_text(qa_pair['question'])

            answer = qa_pair['answer']
            if 'spans' in answer:
                answer['spans'] = [standardize_text(span) for span in answer['spans']]
            if 'validated_answers' in qa_pair:
                for validated_answer in qa_pair['validated_answers']:
                    if 'spans' in answer:
                        validated_answer['spans'] = [standardize_text(span) for span in validated_answer['spans']]
    return dataset


def standardize_text(text):
    # I don't see a reason to differentiate between "No-Break Space" and regular space
    text = text.replace('&#160;', ' ')

    text = html.unescape(text)

    # There is a pattern that repeats itself 97 times in the train set and 16 in the
    # dev set: "<letters>.:<digits>". It originates from the method of parsing the
    # Wikipedia pages. In such an occurrence, "<letters>." is the last word of a
    # sentence, followed by a period. Then, in the wikipedia page, follows a superscript
    # of digits within square brackets, which is a hyperlink to a reference. After the
    # hyperlink there is a colon, ":", followed by <digits>. These digits are the page
    # within the reference.
    # Example: https://en.wikipedia.org/wiki/Polish%E2%80%93Ottoman_War_(1672%E2%80%931676)
    if '.:' in text:
        text = re.sub('\.:d+', '\.', text)
    return text

import itertools
def get_all_subsequences(full_list):

    def contains_sublist(lst, sublst):
        n = len(sublst)
        return any((list(sublst) == lst[i:i+n]) for i in range(len(lst)-n+1))
    
    subsequences = []
    for j in range(len(full_list), 0, -1):
        subsequences.extend([' '.join(part) for part in itertools.combinations(full_list, j) if contains_sublist(full_list, part)])

    return subsequences
