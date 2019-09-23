import itertools
import json
from overrides import overrides
import operator
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

from pytorch_transformers import BasicTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension.util import split_tokens_by_hyphen
from allennlp.data.fields import (Field, TextField, IndexField, LabelField, ListField,
                                  MetadataField, SequenceLabelField, SpanField, ArrayField)
from tqdm import tqdm

from src.nhelpers import *
from src.preprocessing.utils import SPAN_ANSWER_TYPE, SPAN_ANSWER_TYPES, ALL_ANSWER_TYPES, MULTIPLE_SPAN
from src.preprocessing.utils import get_answer_type, fill_token_indices, token_to_span, standardize_dataset

@DatasetReader.register("nabert++")
class NaBertDropReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_pieces: int = 512,
                 max_count: int = 10,
                 max_numbers_expression: int = 2,
                 answer_types: List[str] = None,
                 bio_types: List[str] = None,
                 use_validated: bool = True,
                 wordpiece_numbers: bool = True,
                 number_tokenizer: Tokenizer = None,
                 custom_word_to_num: bool = True,
                 max_depth: int = 3,
                 extra_numbers: List[float] = [],
                 max_instances=-1,
                 uncased: bool = True,
                 is_training: bool = False,
                 target_number_rounding: bool = True,
                 standardize_texts: bool = True,
                 improve_number_extraction: bool = True,
                 discard_impossible_number_questions: bool = True,
                 keep_impossible_number_questions_which_exist_as_spans: bool = False,
                 flexibility_threshold: int = 1000,
                 multispan_allow_all_heads_to_answer: bool = False):
        super(NaBertDropReader, self).__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_pieces = max_pieces
        self.max_count = max_count
        self.max_instances = max_instances
        self.max_numbers_expression = max_numbers_expression
        self.answer_types = answer_types or ALL_ANSWER_TYPES
        self.bio_types = bio_types or [MULTIPLE_SPAN]
        self.use_validated = use_validated
        self.wordpiece_numbers = wordpiece_numbers
        self.number_tokenizer = number_tokenizer or WordTokenizer()
        self.max_depth = max_depth
        self.extra_numbers = extra_numbers
        self.op_dict = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
        self.operations = list(enumerate(self.op_dict.keys()))
        self.templates = [lambda x,y,z: (x + y) * z,
                          lambda x,y,z: (x - y) * z,
                          lambda x,y,z: (x + y) / z,
                          lambda x,y,z: (x - y) / z,
                          lambda x,y,z: x * y / z]
        self.template_strings = ['(%s + %s) * %s',
                                 '(%s - %s) * %s',
                                 '(%s + %s) / %s',
                                 '(%s - %s) / %s',
                                 '%s * %s / %s',]
        if custom_word_to_num:
            self.word_to_num = get_number_from_word
        else:
            self.word_to_num = DropReader.convert_word_to_number

        self._uncased = uncased
        self._is_training = is_training
        self.target_number_rounding = target_number_rounding
        self.standardize_texts = standardize_texts
        self.improve_number_extraction = improve_number_extraction
        self.keep_impossible_number_questions_which_exist_as_spans = \
            keep_impossible_number_questions_which_exist_as_spans
        self.discard_impossible_number_questions = discard_impossible_number_questions
        self.flexibility_threshold = flexibility_threshold
        self.multispan_allow_all_heads_to_answer = multispan_allow_all_heads_to_answer

        self.basic_tokenizer = BasicTokenizer(do_lower_case=uncased)
    
    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, encoding = "utf8") as dataset_file:
            dataset = json.load(dataset_file)

        if self.standardize_texts and self._is_training:
            dataset = standardize_dataset(dataset)

        instances_count = 0
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info["passage"].strip()

            if self.wordpiece_numbers:
                # In this case we actually first use a basic `WordTokenizer`, where each token is
                # additionally split on any hyphen it contains.
                word_tokens = split_tokens_by_hyphen(self.number_tokenizer.tokenize(passage_text))
            else:
                word_tokens = self.tokenizer.tokenize(passage_text)

            # Auxiliary variables for handling numbers from the passage
            numbers_in_passage = []
            number_indices = []
            number_words = []
            number_len = []
            passage_tokens = []
            curr_index = 0

            # Get all passage numbers
            for token in word_tokens:
                # Wordpiece tokenization is done here.
                # In addition, every token recognized as a number is stored for arithmetic processing.
                number = self.word_to_num(token.text, self.improve_number_extraction)
                wordpieces = self.tokenizer.tokenize(token.text)
                num_wordpieces = len(wordpieces)
                if number is not None:
                    numbers_in_passage.append(number)
                    number_indices.append(curr_index)
                    number_words.append(token.text)
                    number_len.append(num_wordpieces)
                passage_tokens += wordpieces
                curr_index += num_wordpieces
            
            passage_tokens = fill_token_indices(passage_tokens, passage_text, self._uncased, self.basic_tokenizer, word_tokens)

            # Process questions from this passage
            for qa_pair in passage_info["qa_pairs"]:
                if 0 < self.max_instances <= instances_count:
                    return

                question_id = qa_pair["query_id"]
                question_text = qa_pair["question"].strip()
                
                answer_annotations: List[Dict] = list()
                specific_answer_type = None
                if 'answer' in qa_pair and qa_pair['answer']:
                    answer = qa_pair['answer']

                    specific_answer_type = get_answer_type(answer)
                    if specific_answer_type not in self.answer_types:
                        continue

                    answer_annotations.append(answer)

                if self.use_validated and "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                    answer_annotations += qa_pair["validated_answers"]

                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 passage_tokens,
                                                 numbers_in_passage,
                                                 number_words,
                                                 number_indices,
                                                 number_len,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 specific_answer_type)
                if instance is not None:
                    instances_count += 1
                    yield instance

    @overrides
    def text_to_instance(self,
                         question_text: str,
                         passage_text: str,
                         passage_tokens: List[Token],
                         numbers_in_passage: List[Any],
                         number_words: List[str],
                         number_indices: List[int],
                         number_len: List[int],
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict] = None,
                         specific_answer_type: str = None) -> Optional[Instance]:
        # Tokenize question and passage
        question_tokens = self.tokenizer.tokenize(question_text)
        question_tokens = fill_token_indices(question_tokens, question_text, self._uncased, self.basic_tokenizer)

        qlen = len(question_tokens)

        qp_tokens = [Token('[CLS]')] + question_tokens + [Token('[SEP]')] + passage_tokens

        # if qp has more than max_pieces tokens (including CLS and SEP), clip the passage
        max_passage_length = -1
        if len(qp_tokens) > self.max_pieces - 1:
            qp_tokens = qp_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - qlen - 3]
            plen = len(passage_tokens)
            number_indices, number_len, numbers_in_passage = \
                clipped_passage_num(number_indices, number_len, numbers_in_passage, plen)
            max_passage_length = token_to_span(passage_tokens[-1])[1] if plen > 0 else 0
        
        qp_tokens += [Token('[SEP]')]
        # update the indices of the numbers with respect to the question.
        # Not done in-place so they won't change the numbers saved for the passage
        number_indices = [index + qlen + 2 for index in number_indices] + [-1]
        number_len = number_len + [1]
        numbers_in_passage = numbers_in_passage + [0]
        number_tokens = [Token(str(number)) for number in numbers_in_passage]
        extra_number_tokens = [Token(str(num)) for num in self.extra_numbers]
        
        mask_indices = [0, qlen + 1, len(qp_tokens) - 1]
        
        fields: Dict[str, Field] = {}
            
        # Add feature fields
        qp_field = TextField(qp_tokens, self.token_indexers)
        fields["question_passage"] = qp_field
       
        number_token_indices = \
            [ArrayField(np.arange(start_ind, start_ind + number_len[i]), padding_value=-1) 
             for i, start_ind in enumerate(number_indices)]
        fields["number_indices"] = ListField(number_token_indices)
        numbers_in_passage_field = TextField(number_tokens, self.token_indexers)
        extra_numbers_field = TextField(extra_number_tokens, self.token_indexers)
        mask_index_fields: List[Field] = [IndexField(index, qp_field) for index in mask_indices]
        fields["mask_indices"] = ListField(mask_index_fields)

        # Compile question, passage, answer metadata
        metadata = {"original_passage": passage_text,
                    "original_question": question_text,
                    "original_numbers": numbers_in_passage,
                    "original_number_words": number_words,
                    "extra_numbers": self.extra_numbers,
                    "passage_tokens": passage_tokens,
                    "question_tokens": question_tokens,
                    "question_passage_tokens": qp_tokens,
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "max_passage_length": max_passage_length}

        # in a word broken up into pieces, every piece except the first should be ignored when calculating the loss
        wordpiece_mask = [not token.text.startswith('##') for token in qp_tokens]
        wordpiece_mask = np.array(wordpiece_mask)
        fields['bio_wordpiece_mask'] = ArrayField(wordpiece_mask, dtype=np.int64)

        if answer_annotations:            
            # Get answer type, answer text, tokenize
            # For multi-span, remove repeating answers. Although possible, in the dataset it is mostly mistakes.
            answer_type, answer_texts = DropReader.extract_answer_info_from_annotation(answer_annotations[0])
            if answer_type == SPAN_ANSWER_TYPE:
                answer_texts = list(OrderedDict.fromkeys(answer_texts))
            tokenized_answer_texts = []
            for answer_text in answer_texts:
                answer_tokens = self.tokenizer.tokenize(answer_text)
                tokenized_answer_text = ' '.join(token.text for token in answer_tokens)
                if tokenized_answer_text not in tokenized_answer_texts:
                    tokenized_answer_texts.append(tokenized_answer_text)

            metadata["answer_annotations"] = answer_annotations
            metadata["answer_texts"] = answer_texts
            metadata["answer_tokens"] = tokenized_answer_texts
            
            # Find answer text in question and passage
            valid_question_spans = DropReader.find_valid_spans(question_tokens, tokenized_answer_texts)
            for span_ind, span in enumerate(valid_question_spans):
                valid_question_spans[span_ind] = (span[0] + 1, span[1] + 1)
            valid_passage_spans = DropReader.find_valid_spans(passage_tokens, tokenized_answer_texts)
            for span_ind, span in enumerate(valid_passage_spans):
                valid_passage_spans[span_ind] = (span[0] + qlen + 2, span[1] + qlen + 2)

            # throw away an instance in training if a span appearing in the answer is missing from the question and passage
            if self._is_training:
                if specific_answer_type in SPAN_ANSWER_TYPES:
                    for tokenized_answer_text in tokenized_answer_texts:
                        temp_spans = DropReader.find_valid_spans(qp_field, [tokenized_answer_text])
                        if len(temp_spans) == 0:
                            return None

            # Get target numbers
            target_numbers = []
            if specific_answer_type != MULTIPLE_SPAN or self.multispan_allow_all_heads_to_answer:
                for answer_text in answer_texts:
                    number = self.word_to_num(answer_text, self.improve_number_extraction)
                    if number is not None:
                        target_numbers.append(number)
            
            # Get possible ways to arrive at target numbers with add/sub
            valid_expressions: List[List[int]] = []
            exp_strings = None
            if answer_type in ["number", "date"]:
                if self.target_number_rounding:
                    valid_expressions = \
                        find_valid_add_sub_expressions_with_rounding(
                            self.extra_numbers + numbers_in_passage,
                            target_numbers,
                            self.max_numbers_expression)
                else:
                    valid_expressions = \
                        DropReader.find_valid_add_sub_expressions(self.extra_numbers + numbers_in_passage,
                                                                  target_numbers,
                                                                  self.max_numbers_expression)

                if self.discard_impossible_number_questions:
                    # The train set was verified to have all of its target_numbers lists of length 1.
                    if (answer_type == "number" and
                            len(valid_expressions) == 0 and
                            self._is_training and
                            self.max_count < target_numbers[0]):
                        # The number to predict can't be derived from any head, so we shouldn't train on it.
                        # arithmetic - no expressions that yield the number to predict.
                        # counting - the maximal count is smaller than the number to predict.

                        # However, although the answer is marked in the dataset as a number type answer,
                        # maybe it cannot be found due to a bug in DROP's text parsing.
                        # So in addition, we try to find the answer as a span in the text.
                        # If the answer is indeed a span in the text, we don't discard that question.
                        if len(valid_question_spans) == 0 and len(valid_passage_spans) == 0:
                            return None
                        if not self.keep_impossible_number_questions_which_exist_as_spans:
                            return None

            # Get possible ways to arrive at target numbers with counting
            valid_counts: List[int] = []
            if answer_type in ["number"]:
                numbers_for_count = list(range(self.max_count + 1))
                valid_counts = DropReader.find_valid_counts(numbers_for_count, target_numbers)
            
            # Update metadata with answer info
            answer_info = {"answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "expressions": valid_expressions,
                           "counts": valid_counts}
            metadata["answer_info"] = answer_info
        
            # Add answer fields
            passage_span_fields: List[Field] = []
            if specific_answer_type != MULTIPLE_SPAN or self.multispan_allow_all_heads_to_answer:
                passage_span_fields: List[Field] = [SpanField(span[0], span[1], qp_field) for span in valid_passage_spans]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, qp_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields: List[Field] = []
            if specific_answer_type != MULTIPLE_SPAN or self.multispan_allow_all_heads_to_answer:
                question_span_fields: List[Field] = [SpanField(span[0], span[1], qp_field) for span in valid_question_spans]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, qp_field))
            fields["answer_as_question_spans"] = ListField(question_span_fields)
            
            add_sub_signs_field: List[Field] = []
            extra_signs_field: List[Field] = []
            for signs_for_one_add_sub_expressions in valid_expressions:
                extra_signs = signs_for_one_add_sub_expressions[:len(self.extra_numbers)]
                normal_signs = signs_for_one_add_sub_expressions[len(self.extra_numbers):]
                add_sub_signs_field.append(SequenceLabelField(normal_signs, numbers_in_passage_field))
                extra_signs_field.append(SequenceLabelField(extra_signs, extra_numbers_field))
            if not add_sub_signs_field:
                add_sub_signs_field.append(SequenceLabelField([0] * len(number_tokens), numbers_in_passage_field))
            if not extra_signs_field:
                extra_signs_field.append(SequenceLabelField([0] * len(self.extra_numbers), extra_numbers_field))
            fields["answer_as_expressions"] = ListField(add_sub_signs_field)
            if self.extra_numbers:
                fields["answer_as_expressions_extra"] = ListField(extra_signs_field)

            count_fields: List[Field] = [LabelField(count_label, skip_indexing=True) for count_label in valid_counts]
            if not count_fields:
                count_fields.append(LabelField(-1, skip_indexing=True))
            fields["answer_as_counts"] = ListField(count_fields)
            
            no_answer_bios = SequenceLabelField([0] * len(qp_tokens), sequence_field=qp_field)
            if (specific_answer_type in self.bio_types) and (len(valid_passage_spans) > 0 or len(valid_question_spans) > 0):
                
                # Used for flexible BIO loss
                # START
                
                spans_dict = {}
                text_to_disjoint_bios: List[ListField] = []
                flexibility_count = 1
                for tokenized_answer_text in tokenized_answer_texts:
                    spans = DropReader.find_valid_spans(qp_tokens, [tokenized_answer_text])
                    if len(spans) == 0:
                        # possible if the passage was clipped, but not for all of the answers
                        continue
                    spans_dict[tokenized_answer_text] = spans

                    disjoint_bios: List[SequenceLabelField] = []
                    for span_ind, span in enumerate(spans):
                        bios = create_bio_labels([span], len(qp_field))
                        disjoint_bios.append(SequenceLabelField(bios, sequence_field=qp_field))

                    text_to_disjoint_bios.append(ListField(disjoint_bios))
                    flexibility_count *= ((2**len(spans)) - 1)

                fields["answer_as_text_to_disjoint_bios"] = ListField(text_to_disjoint_bios)

                if (flexibility_count < self.flexibility_threshold):
                    # generate all non-empty span combinations per each text
                    spans_combinations_dict = {}
                    for key, spans in spans_dict.items():
                        spans_combinations_dict[key] = all_combinations = []
                        for i in range(1, len(spans) + 1):
                            all_combinations += list(itertools.combinations(spans, i))

                    # calculate product between all the combinations per each text
                    packed_gold_spans_list = itertools.product(*list(spans_combinations_dict.values()))
                    bios_list: List[SequenceLabelField] = []
                    for packed_gold_spans in packed_gold_spans_list:
                        gold_spans = [s for sublist in packed_gold_spans for s in sublist]
                        bios = create_bio_labels(gold_spans, len(qp_field))
                        bios_list.append(SequenceLabelField(bios, sequence_field=qp_field))
                    
                    fields["answer_as_list_of_bios"] = ListField(bios_list)
                    fields["answer_as_text_to_disjoint_bios"] = ListField([ListField([no_answer_bios])])
                else:
                    fields["answer_as_list_of_bios"] = ListField([no_answer_bios])

                # END

                # Used for both "require-all" BIO loss and flexible loss
                bio_labels = create_bio_labels(valid_question_spans + valid_passage_spans, len(qp_field))
                fields['span_bio_labels'] = SequenceLabelField(bio_labels, sequence_field=qp_field)

                fields["is_bio_mask"] = LabelField(1, skip_indexing=True)
            else:
                fields["answer_as_text_to_disjoint_bios"] = ListField([ListField([no_answer_bios])])
                fields["answer_as_list_of_bios"] = ListField([no_answer_bios])

                # create all 'O' BIO labels for non-span questions
                fields['span_bio_labels'] = no_answer_bios
                fields["is_bio_mask"] = LabelField(0, skip_indexing=True)

        fields["metadata"] = MetadataField(metadata)
        
        return Instance(fields)


def create_bio_labels(spans: List[Tuple[int, int]], n_labels: int):

    # initialize all labels to O
    labels = [0] * n_labels

    for span in spans:
        start = span[0]
        end = span[1]
        # create B labels
        labels[start] = 1
        # create I labels
        labels[start+1:end+1] = [2] * (end - start)

    return labels


def find_valid_add_sub_expressions_with_rounding(
        numbers: List[int],
        targets: List[int],
        max_number_of_numbers_to_consider: int = 2) -> List[List[int]]:
    valid_signs_for_add_sub_expressions = []
    # TODO: Try smaller numbers?
    for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
        possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
        for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
            indices = [it[0] for it in number_combination]
            values = [it[1] for it in number_combination]
            for signs in possible_signs:
                eval_value = sum(sign * value for sign, value in zip(signs, values))
                # our added rounding, our only change compared to `find_valid_add_sub_expressions`
                eval_value = round(eval_value, 5)
                # end of our added rounding
                if eval_value in targets:
                    labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                    for index, sign in zip(indices, signs):
                        labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                    valid_signs_for_add_sub_expressions.append(labels_for_numbers)
    return valid_signs_for_add_sub_expressions
