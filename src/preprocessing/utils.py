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

def fill_token_indices(tokens, full_text, uncased, basic_tokenizer, word_tokens=None):
    if uncased:
        full_text = full_text.lower()

    new_tokens = []
    has_unknowns = False

    temp_text = full_text
    reconstructed_full_text = ''

    absolute_index = 0
    for i, token in enumerate(tokens):
        token_text = token.text
        token_text_to_search = token_text[2:] if len(token_text) > 2 and token_text[:2] == "##" else token_text
        
        if token_text == '[UNK]':
            new_tokens.append(Token(text = token_text, lemma_ = token_text, idx=absolute_index)) 
            # lemma as placeholder, index to search from later
            has_unknowns = True
            continue

        relative_index = basic_tokenizer._run_strip_accents(temp_text).find(token_text_to_search)

        start_idx = absolute_index + relative_index
        end_idx = start_idx + len(token_text_to_search) # exclusive
        absolute_index = end_idx
        token_source_text = full_text[start_idx : end_idx]

        first_part = temp_text[:relative_index]
        second_part = token_source_text
        reconstructed_full_text += first_part + second_part
        temp_text = temp_text[relative_index + len(token_source_text):]

        new_tokens.append(Token(text = token_text, lemma_ = token_source_text, idx = start_idx))

    if has_unknowns:
        reconstructed_full_text = ''
        word_tokens_text = ' '.join([word_token.text for word_token in word_tokens]) if word_tokens is not None else full_text
        basic_tokens, j, constructed_token = basic_tokenizer.tokenize(word_tokens_text), 0, ''
        padding_idx = 0
        for i, token in enumerate(new_tokens):
            if token.text != '[UNK]':
                constructed_token += token.lemma_
                if constructed_token == basic_tokens[j]:
                    constructed_token = ''
                    j += 1
            else:
                relative_index = basic_tokenizer._run_strip_accents(full_text[token.idx:]).find(basic_tokens[j])
                new_tokens[i] = Token(text = token.text, lemma_ = basic_tokens[j], idx = token.idx + relative_index)
                j += 1
            
            padding = full_text[padding_idx : new_tokens[i].idx]
            reconstructed_full_text += padding + full_text[new_tokens[i].idx : new_tokens[i].idx + len(new_tokens[i].lemma_)]
            padding_idx = new_tokens[i].idx + len(new_tokens[i].lemma_)
    
    # Will happen in very rare cases due to accents stripping changing the length of the word
    #if reconstructed_full_text != full_text:
    #    raise Exception('Error with token indices')
    
    return new_tokens

def token_to_span(token):
    start = token.idx
    end = token.idx + len(token.lemma_)
    return (start, end)


def standardize_dataset(dataset):
    for passage_info in dataset.values():
        passage_info['passage'] = standardize_text(passage_info['passage'])
        for qa_pair in passage_info["qa_pairs"]:
            qa_pair['question'] = standardize_text(qa_pair['question'])

            answer = qa_pair['answer']
            if 'answer' in qa_pair:
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
