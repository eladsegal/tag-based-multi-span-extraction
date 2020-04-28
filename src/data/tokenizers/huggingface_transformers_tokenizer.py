from typing import Dict, List, Union, Tuple, Any
from overrides import overrides

from allennlp.data.tokenizers import Token, Tokenizer

from transformers.tokenization_auto import AutoTokenizer

from src.data.tokenizers.tokenization_utils import (tokenize_with_offsets, 
                                                    token_offsets_to_strings, 
                                                    align_tokens_to_tokens, 
                                                    alignment_to_wordpieces_list,
                                                    wordpieces_list_to_token_wordpieces)

@Tokenizer.register("huggingface_transformers")
class HuggingfaceTransformersTokenizer(Tokenizer):
    def __init__(self, pretrained_model: str):
        self._pretrained_model = pretrained_model

        self._init_kwargs = {}
        self._kwargs = {}

        if pretrained_model.startswith('roberta-'):
            self._kwargs['add_prefix_space'] = True

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **self._init_kwargs)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(text=token) for token in self._tokenizer.tokenize(text, **self._kwargs)]

    def tokenize_with_offsets(self, text: str) -> List[Token]:
        tokens, offsets = tokenize_with_offsets(self._tokenizer.tokenize, text, **self._kwargs)

        tokens_with_offsets = []
        for i, offset in enumerate(offsets):
            if i < len(offsets) - 1:
                next_offset = offsets[i + 1]
                original_token_text = text[offset:next_offset]
            else:
                original_token_text = text[offset:]
            original_token_text = original_token_text.strip()
            tokens_with_offsets.append(Token(text=tokens[i], idx=offset, lemma_=original_token_text))

        return tokens_with_offsets

    @property
    def encode_plus(self):
        return self._tokenizer.encode_plus

    def convert_to_tokens(self, encoded_inputs, type_groups, amend_cutoff_wordpieces=False):
        text_tokens = self._tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'])
        token_type_ids = encoded_inputs['token_type_ids']
        special_tokens_mask = encoded_inputs['special_tokens_mask']
        
        tokens = []
        question_token_index = 0
        passage_token_index = 0
        type_token_indexes = [0] * len(type_groups)
        for i, text_token in enumerate(text_tokens):
            type_index = token_type_ids[i]
            if special_tokens_mask[i] == 1:
                token = Token(text=text_token, type_id=type_index)
            elif type_index < len(type_groups):
                type_tokens = type_groups[type_index]['tokens']
                token_index = type_token_indexes[type_index]

                token = Token(text=text_token, 
                                idx=type_tokens[token_index].idx, 
                                lemma_=type_tokens[token_index].lemma_, 
                                type_id=type_index)
                type_token_indexes[type_index] += 1
            else:
                token = Token(text=text_token, 
                                type_id=type_index)
            tokens.append(token)

        if amend_cutoff_wordpieces:
            for type_index, num_of_tokens in enumerate(type_token_indexes):
                type_tokens = type_groups[type_index]['tokens']
                if len(type_tokens) > num_of_tokens:
                    last_index = num_of_tokens - 1
                    type_wordpieces = type_groups[type_index]['wordpieces']
                    last_word = type_wordpieces[last_index]
                    if last_word[-1] != last_index:
                        amended_lemma = ''.join([type_tokens[token_index].lemma_ 
                                                 for token_index 
                                                 in last_word[last_word.index(last_index):]])
                        position = type_groups[type_index]['position']
                        last_token = type_tokens[last_index]
                        tokens[position + last_index] = Token(text=last_token.text, 
                                                                idx=last_token.idx,
                                                                lemma_=amended_lemma,
                                                                type_id=type_index)
        return tokens, type_token_indexes

    @staticmethod
    def align_tokens_to_tokens(text, words, tokens):
        from_strings = token_offsets_to_strings([token.idx for token in words], text)
        to_strings = token_offsets_to_strings([token.idx for token in tokens], text)
        alignment = align_tokens_to_tokens(from_strings, to_strings)
        return alignment

    @staticmethod
    def alignment_to_token_wordpieces(alignment):
        return wordpieces_list_to_token_wordpieces(alignment_to_wordpieces_list(alignment))

    @staticmethod
    def get_type_position_in_sequence(seq_type_id, token_type_ids, special_tokens_mask):
        # index of the first token of a valid seq_type_id
        position = next(i for i, type_id in enumerate(token_type_ids) 
                    if type_id == seq_type_id and special_tokens_mask[i] != 1)
        return position

    @staticmethod
    def adjust_wordpieces(type_groups, tokens):
        wordpieces = []
        for type_group in type_groups:
            type_position = type_group['position']
            type_wordpieces = type_group['wordpieces']
            wordpieces += [[len(wordpieces) + i] for i in range(type_position - len(wordpieces))]
            wordpieces += HuggingfaceTransformersTokenizer._shift_wordpieces(type_wordpieces, 
                                                                             type_position, 
                                                                             type_group['num_of_tokens'])
        wordpieces += [[len(wordpieces) + i] for i in range(len(tokens) - len(wordpieces))]
        return wordpieces

    @staticmethod
    def _shift_wordpieces(wordpieces, position, num_of_tokens):
        return [[wordpiece + position for wordpiece in word 
                if wordpiece < num_of_tokens] for word in wordpieces 
                if word[0] < num_of_tokens][:num_of_tokens]