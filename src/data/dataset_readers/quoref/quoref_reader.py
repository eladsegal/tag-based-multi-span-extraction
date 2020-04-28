import logging

import numpy as np
import json
from overrides import overrides
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, MetadataField
from tqdm import tqdm

from src.data.dataset_readers.drop.drop_utils import (AnswerType, ALL_ANSWER_TYPES, get_answer_type, 
                                                      standardize_dataset, extract_answer_info_from_annotation)
from src.data.dataset_readers.utils import standardize_text_simple, standardize_text_advanced
from src.data.dataset_readers.utils import custom_word_tokenizer, split_tokens_by_hyphen, index_text_to_tokens
from src.data.dataset_readers.utils import is_pickle_dict_valid, load_pkl, save_pkl
from src.data.dataset_readers.answer_field_generators.answer_field_generator import AnswerFieldGenerator
from src.data.fields.labels_field import LabelsField

logger = logging.getLogger(__name__)

@DatasetReader.register('tbmse_quoref')
class QuorefReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 answer_field_generators: Dict[str, AnswerFieldGenerator],
                 answer_generator_names_per_type: Dict[str, List[str]],
                 old_reader_behavior: bool,
                 lazy: bool = False,
                 is_training: bool = False,
                 max_instances = -1,
                 answer_types_filter: List[str] = ALL_ANSWER_TYPES,
                 max_pieces: int = 512,
                 uncased: bool = False,
                 standardize_texts: bool = False, # since we need to use answer_start, we can't modify the text without tracking the changes,
                 pickle: Dict[str, Any] = {'action': None}):
        super().__init__(lazy)
        self._lazy = lazy

        self._tokenizer = tokenizer

        self._answer_field_generators = answer_field_generators
        self._answer_generator_names_per_type = answer_generator_names_per_type

        self._old_reader_behavior = old_reader_behavior

        self._is_training = is_training
        self._max_instances = max_instances
        
        self._answer_types_filter = answer_types_filter

        self._max_pieces = max_pieces
        self._uncased = uncased

        self._standardize_text_func = (standardize_text_advanced if standardize_texts
                                       else standardize_text_simple)

        self._pickle = pickle
        if not is_pickle_dict_valid(self._pickle):
            self._pickle['action'] = None

        word_tokenizer = custom_word_tokenizer()
        self._word_tokenize =\
            lambda text: [token for token in split_tokens_by_hyphen(word_tokenizer.tokenize(text))]

    @overrides
    def _read(self, file_path: str):
        if not self._lazy and self._pickle['action'] == 'load':
            # Try to load the data, if it fails then read it from scratch and save it
            loaded_pkl = load_pkl(self._pickle, self._is_training)
            if loaded_pkl is not None:
                for instance in loaded_pkl:
                    yield instance
                return
            else:
                self._pickle['action'] = 'save'

        file_path = cached_path(file_path)
        with open(file_path, encoding = 'utf8') as dataset_file:
            dataset = json.load(dataset_file)

        dataset = standardize_dataset(dataset, self._standardize_text_func)

        global_index = 0
        instances_count = 0
        instances = []
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info['passage']

            # Tokenize passage
            passage_tokens = self._tokenizer.tokenize_with_offsets(passage_text)
            passage_text_index_to_token_index = index_text_to_tokens(passage_text, passage_tokens)
            passage_words = self._word_tokenize(passage_text)
            passage_alignment = self._tokenizer.align_tokens_to_tokens(passage_text, passage_words, passage_tokens)
            passage_wordpieces = self._tokenizer.alignment_to_token_wordpieces(passage_alignment)

            # Process questions from this passage
            for relative_index, qa_pair in enumerate(passage_info['qa_pairs']):
                if 0 < self._max_instances <= instances_count:
                    if not self._lazy and self._pickle['action'] == 'save':
                        save_pkl(instances, self._pickle, self._is_training)
                    return

                question_id = qa_pair['query_id']
                question_text = qa_pair['question']

                answer_annotations: List[Dict] = list()
                original_answer_annotations: List[List[Dict]] = list()
                answer_type = None
                if 'answer' in qa_pair and qa_pair['answer']:
                    answer = qa_pair['answer']
                    original_answer = qa_pair['original_answer']

                    answer_type = get_answer_type(answer)
                    if answer_type is None or answer_type not in self._answer_types_filter:
                        continue

                    answer_annotations.append(answer)
                    original_answer_annotations.append(original_answer)

                    # If the standardization deleted characters then we need to adjust answer_start
                    deletion_indexes = self._standardize_text_func(passage_info['original_passage'], deletions_tracking=True)[1]
                    for span in original_answer:
                        answer_start = span['answer_start']
                        for index in deletion_indexes.keys():
                            if span['answer_start'] > index:
                                answer_start -= deletion_indexes[index]
                        span['answer_start'] = answer_start

                if self._is_training and answer_type is None:
                    continue

                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 passage_tokens,
                                                 passage_text_index_to_token_index,
                                                 passage_wordpieces,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 original_answer_annotations,
                                                 answer_type,
                                                 global_index + relative_index)
                if instance is not None:
                    instances_count += 1
                    if not self._lazy:
                        instances.append(instance)
                    yield instance
            global_index += len(passage_info['qa_pairs'])
        if not self._lazy and self._pickle['action'] == 'save':
            save_pkl(instances, self._pickle, self._is_training)

    @overrides
    def text_to_instance(self,
                         question_text: str,
                         passage_text: str,
                         passage_tokens: List[Token],
                         passage_text_index_to_token_index: List[int],
                         passage_wordpieces: List[List[int]],
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict] = None,
                         original_answer_annotations: List[List[Dict]] = None,
                         answer_type: str = None,
                         instance_index: int = None) -> Optional[Instance]:

        # Tokenize question
        question_tokens = self._tokenizer.tokenize_with_offsets(question_text)
        question_text_index_to_token_index = index_text_to_tokens(question_text, question_tokens)
        question_words = self._word_tokenize(question_text)
        question_alignment = self._tokenizer.align_tokens_to_tokens(question_text, question_words, question_tokens)
        question_wordpieces = self._tokenizer.alignment_to_token_wordpieces(question_alignment)

        # Index tokens
        encoded_inputs = self._tokenizer.encode_plus([token.text for token in question_tokens], [token.text for token in passage_tokens], 
                                    add_special_tokens=True, max_length=self._max_pieces, 
                                    truncation_strategy='only_second',
                                    return_token_type_ids=True,
                                    return_special_tokens_mask=True)
        question_passage_token_type_ids = encoded_inputs['token_type_ids']
        question_passage_special_tokens_mask = encoded_inputs['special_tokens_mask']

        question_position = self._tokenizer.get_type_position_in_sequence(0, question_passage_token_type_ids, question_passage_special_tokens_mask)
        passage_position = self._tokenizer.get_type_position_in_sequence(1, question_passage_token_type_ids, question_passage_special_tokens_mask)
        question_passage_tokens, num_of_tokens_per_type = self._tokenizer.convert_to_tokens(encoded_inputs, [
            {'tokens': question_tokens, 'wordpieces': question_wordpieces, 'position': question_position}, 
            {'tokens': passage_tokens, 'wordpieces': passage_wordpieces, 'position': passage_position}
        ])

        # Adjust wordpieces
        question_passage_wordpieces = self._tokenizer.adjust_wordpieces([
            {'wordpieces': question_wordpieces, 'position': question_position, 'num_of_tokens': num_of_tokens_per_type[0]}, 
            {'wordpieces': passage_wordpieces, 'position': passage_position, 'num_of_tokens': num_of_tokens_per_type[1]}
        ], question_passage_tokens)

        # Adjust text index to token index
        question_text_index_to_token_index = [token_index + question_position for i, token_index in enumerate(question_text_index_to_token_index)
                                              if token_index < num_of_tokens_per_type[0]]
        passage_text_index_to_token_index = [token_index + passage_position for i, token_index in enumerate(passage_text_index_to_token_index)
                                              if token_index < num_of_tokens_per_type[1]]

        # Truncation-related code
        encoded_passage_tokens_length = num_of_tokens_per_type[1]
        if encoded_passage_tokens_length > 0:
            if encoded_passage_tokens_length < len(passage_tokens):
                first_truncated_passage_token = passage_tokens[encoded_passage_tokens_length]
                max_passage_length = first_truncated_passage_token.idx
            else:
                max_passage_length = -1
        else:
            max_passage_length = 0


        fields: Dict[str, Field] = {}

        fields['question_passage_tokens'] = question_passage_field = LabelsField(encoded_inputs['input_ids'])
        fields['question_passage_token_type_ids'] = LabelsField(question_passage_token_type_ids)
        fields['question_passage_special_tokens_mask'] = LabelsField(question_passage_special_tokens_mask)
        fields['question_passage_pad_mask'] = LabelsField([1] * len(question_passage_tokens))

        # in a word broken up into pieces, every piece except the first should be ignored when calculating the loss
        first_wordpiece_mask = [i == wordpieces[0] for i, wordpieces in enumerate(question_passage_wordpieces)]
        fields['first_wordpiece_mask'] = LabelsField(first_wordpiece_mask)

        # Compile question, passage, answer metadata
        metadata = {'original_passage': passage_text,
                    'original_question': question_text,
                    'passage_tokens': passage_tokens,
                    'question_tokens': question_tokens,
                    'question_passage_tokens': question_passage_tokens,
                    'question_passage_wordpieces': question_passage_wordpieces,
                    'passage_id': passage_id,
                    'question_id': question_id,
                    'max_passage_length': max_passage_length}
        if instance_index is not None:
            metadata['instance_index'] = instance_index

        if answer_annotations:
            _, answer_texts = extract_answer_info_from_annotation(answer_annotations[0])
            answer_texts = list(OrderedDict.fromkeys(answer_texts))

            gold_indexes = {'question': [], 'passage': []}
            for original_answer_annotation in original_answer_annotations[0]:
                if original_answer_annotation['text'] in answer_texts:
                    gold_index = original_answer_annotation['answer_start']
                    if gold_index not in gold_indexes['passage']:
                        gold_indexes['passage'].append(gold_index)

            metadata['answer_annotations'] = answer_annotations

            kwargs = {
                'seq_tokens': question_passage_tokens,
                'seq_field': question_passage_field,
                'seq_wordpieces': question_passage_wordpieces,
                'question_text': question_text,
                'question_text_index_to_token_index': question_text_index_to_token_index,
                'passage_text': passage_text[:max_passage_length] if max_passage_length > -1 else passage_text,
                'passage_text_index_to_token_index': passage_text_index_to_token_index,
                'answer_texts': answer_texts,
                'gold_indexes': gold_indexes,
                'answer_type': answer_type, # TODO: Elad - Probably temporary, used to mimic the old reader's behavior
                'is_training': self._is_training, # TODO: Elad - Probably temporary, used to mimic the old reader's behavior
                'old_reader_behavior': self._old_reader_behavior # TODO: Elad - temporary, used to mimic the old reader's behavior
            }

            answer_generator_names = None
            if self._answer_generator_names_per_type is not None:
                answer_generator_names = self._answer_generator_names_per_type[answer_type]

            has_answer = False
            for answer_generator_name, answer_field_generator in self._answer_field_generators.items():
                if answer_generator_names is None or answer_generator_name in answer_generator_names:
                    answer_fields, generator_has_answer = answer_field_generator.get_answer_fields(**kwargs)
                    fields.update(answer_fields)
                    has_answer |= generator_has_answer
                else:
                    fields.update(answer_field_generator.get_empty_answer_fields(**kwargs))

            # throw away instances without possible answer generation
            if self._is_training and not has_answer:
                return None

        fields['metadata'] = MetadataField(metadata)
        
        return Instance(fields)
