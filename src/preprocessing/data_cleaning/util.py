from src.preprocessing.data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.tokenizers import Token
import string
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any
from allennlp.data.dataset_readers.reading_comprehension.util import (IGNORED_TOKENS,
                                                                      STRIPPED_CHARACTERS)

def find_valid_spans(passage_tokens: List[Token],
                        answer_texts: List[str]) -> List[Tuple[int, int]]:
    normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
    word_positions: Dict[str, List[int]] = defaultdict(list)
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i)
    spans = []
    for answer_text in answer_texts:
        answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        answer_tokens = split_tokens_by_hyphen(answer_tokens)
        #answer_tokens = [answer_token.split('-') for answer_token in answer_tokens]
        #answer_tokens = [t for tokens in answer_tokens for t in tokens]
        num_answer_tokens = len(answer_tokens)
        if answer_tokens[0] not in word_positions:
            continue
        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index:
                spans.append((span_start, span_end))
    return spans

def split_tokens_by_hyphen(tokens: List[str]) -> List[str]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[str] = []

    for i in range(len(tokens)):
        token = tokens[i]
        if any(hyphen in token for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[str] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token:
                        split_tokens_temp = split_token_by_delimiter(unsplit_token, hyphen)
                        if split_tokens_temp[-1].isdigit():
                            split_tokens += split_tokens_temp
                        else:
                            split_tokens.append(unsplit_token)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens

def split_token_by_delimiter(token: str, delimiter: str) -> List[Token]:
    split_tokens = []
    
    for sub_str in token.split(delimiter):
        if sub_str:
            split_tokens.append(sub_str)
            
        split_tokens.append(delimiter)
        
    if split_tokens:
        split_tokens.pop(-1)
        
        return split_tokens
    else:
        return [token]
