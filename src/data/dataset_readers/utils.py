from typing import List, Tuple, Dict
import html
import re
import sys
import os
import pickle
import unicodedata

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.dataset_readers.reading_comprehension.util import split_tokens_by_hyphen, STRIPPED_CHARACTERS

whitespaces = re.findall(r'\s', u''.join(chr(c) for c in range(sys.maxunicode+1)), re.UNICODE)
empty_chars = ['\u200b', '\ufeff', '\u2061'] # zero width space, byte order mark
def standardize_text_simple(text, deletions_tracking=False):
    for whitespace in whitespaces:
        text = text.replace(whitespace, ' ')

    if deletions_tracking:
        deletion_indexes = {}
        track_deletions(text, deletion_indexes)

    # This is a must for proper tokenization with offsets
    for empty_char in empty_chars:
        text = text.replace(empty_char, '')

    text = ' '.join(text.split()) # use ' ' for all spaces and replace sequence of spaces with single space

    return text if not deletions_tracking else (text, deletion_indexes)

def track_deletions(text, deletion_indexes):
    """
    Track deletions for empty_chars removal and space sequences trimming
    """
    for empty_char in empty_chars:
        for i, char in enumerate(text):
            if char == empty_char:
                deletion_indexes[i] = 1

    initial_space_length = len(text) - len(text.lstrip())
    if initial_space_length > 0:
        deletion_indexes[0] = initial_space_length
    space_sequence = False
    length = 0
    for i, char in enumerate(text):
        if char == ' ':
            space_sequence = True
            length += 1
            if i == len(text) - 1:
                deletion_indexes[i - length] = length
        else:
            if space_sequence:
                if length > 1 and (i - length) > 0:
                    deletion_indexes[i - length] = length - 1
            space_sequence = False
            length = 0

def standardize_text_advanced(text, deletions_tracking=False):
    text = html.unescape(text)
    text = standardize_text_simple(text)

    # There is a pattern that repeats itself 97 times in the train set and 16 in the
    # dev set: "<letters>.:<digits>". It originates from the method of parsing the
    # Wikipedia pages. In such an occurrence, "<letters>." is the last word of a
    # sentence, followed by a period. Then, in the wikipedia page, follows a superscript
    # of digits within square brackets, which is a hyperlink to a reference. After the
    # hyperlink there is a colon, ":", followed by <digits>. These digits are the page
    # within the reference.
    # Example: https://en.wikipedia.org/wiki/Polish%E2%80%93Ottoman_War_(1672%E2%80%931676)
    if '.:' in text:
        text = re.sub('\.:\d+(-\d+)*', '.', text)

    # In a few cases the passage starts with a location and weather description. 
    # Example: "at Lincoln Financial Field, Philadelphia|weather= 60&#160;&#176;F (Clear)".
    # Relevant for 3 passages (15 questions) in the training set and 1 passage (25 questions) in the dev set.
    text.replace("|weather", " weather")

    return text if not deletions_tracking else (text, {})

def custom_word_tokenizer():
    #tokenizer_exceptions = English().Defaults.tokenizer_exceptions
    word_tokenizer = SpacyTokenizer()
    word_tokenizer.spacy.tokenizer = Tokenizer(vocab=word_tokenizer.spacy.tokenizer.vocab, rules={}, 
                                               prefix_search=word_tokenizer.spacy.tokenizer.prefix_search, 
                                               suffix_search=word_tokenizer.spacy.tokenizer.suffix_search, 
                                               infix_finditer=word_tokenizer.spacy.tokenizer.infix_finditer)
    return word_tokenizer

def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    """
    From allennlp's reading_comprehension.util
    """
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]

def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    """
    From allennlp's reading_comprehension.util
    """
    hyphens = ["-", "–", "~", "—"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens

def run_strip_accents(text):
    """
    From tokenization_bert.py by huggingface/transformers.
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def find_all(substr, text):
    matches = []
    start = 0
    while True:
        start = text.find(substr, start)
        if start == -1:
            break
        matches.append(start)
        start += 1
    return matches

def index_text_to_tokens(text, tokens):
    text_index_to_token_index = []
    token_index = 0
    next_token_index = token_index + 1
    index = tokens[token_index].idx
    next_index = tokens[next_token_index].idx
    for i in range(len(text)):
        while True:
            while next_index == index and next_token_index < len(tokens) - 1:
                next_token_index += 1
                next_index = tokens[next_token_index].idx
            if next_index == index and next_token_index == len(tokens) - 1:
                next_token_index += 1
                next_index = len(text)
            
            if i >= index and i < next_index:
                text_index_to_token_index.append(token_index)
                break
            else:
                token_index = next_token_index
                index = next_index

                if next_token_index < len(tokens) - 1:
                    next_token_index += 1
                    next_index = tokens[next_token_index].idx
                else:
                    next_token_index += 1
                    next_index = len(text)
                if (next_token_index > len(tokens)):
                    raise Exception("Error in " + text)
    return text_index_to_token_index

def find_valid_spans(text: str, answer_texts: List[str], 
                     text_index_to_token_index: List[int], 
                     tokens: List[Token], wordpieces: List[List[int]],
                     gold_indexes: List[int]) -> List[Tuple[int, int]]:
    text = text.lower()
    answer_texts_set = set()
    for answer_text in answer_texts:
        option1 = answer_text.lower()
        option2 = option1.strip(STRIPPED_CHARACTERS) 
        option3 = run_strip_accents(option1)
        option4 = run_strip_accents(option2)
        answer_texts_set.update([option1, option2, option3, option4])

    gold_token_indexes = None
    if gold_indexes is not None:
        gold_token_indexes = []
        for gold_index in gold_indexes:
            if gold_index < len(text_index_to_token_index): # if the gold index was not truncated
                if text[gold_index] == ' ' and gold_index < len(text_index_to_token_index) - 1:
                    gold_index += 1
                gold_token_indexes.append(text_index_to_token_index[gold_index])

    valid_spans = set()
    for answer_text in answer_texts_set:
        start_indexes = find_all(answer_text, text)
        for start_index in start_indexes:
            start_token_index = text_index_to_token_index[start_index]
            end_token_index = text_index_to_token_index[start_index + len(answer_text) - 1]

            wordpieces_condition = (wordpieces[start_token_index][0] == start_token_index and 
                                    wordpieces[end_token_index][-1] == end_token_index)

            stripped_answer_text = answer_text.strip(STRIPPED_CHARACTERS)
            stripped_first_token = tokens[start_token_index].lemma_.lower().strip(STRIPPED_CHARACTERS)
            stripped_last_token = tokens[end_token_index].lemma_.lower().strip(STRIPPED_CHARACTERS)
            text_match_condition = (stripped_answer_text.startswith(stripped_first_token) and 
                                        stripped_answer_text.endswith(stripped_last_token))

            gold_index_condition = gold_token_indexes is None or start_token_index in gold_token_indexes

            if wordpieces_condition and text_match_condition and gold_index_condition:
                valid_spans.add((start_token_index, end_token_index))

    return valid_spans

def save_pkl(instances, pickle_dict, is_training):
    if is_pickle_dict_valid(pickle_dict):
        pkl_path = get_pkl_path(pickle_dict, is_training)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as dataset_file:
            pickle.dump(instances, dataset_file)

def load_pkl(pickle_dict, is_training):
    try:
        with open(get_pkl_path(pickle_dict, is_training), 'rb') as dataset_pkl:
            return pickle.load(dataset_pkl)
    except Exception as e:
        return None

def get_pkl_path(pickle_dict, is_training):
    return os.path.join(pickle_dict['path'], f"{pickle_dict['file_name']}_{'train' if is_training else 'dev'}.pkl")

def is_pickle_dict_valid(pickle_dict):
    return pickle_dict is not None and 'path' in pickle_dict and 'file_name' in pickle_dict