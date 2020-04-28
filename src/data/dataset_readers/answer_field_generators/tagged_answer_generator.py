from typing import Dict, List, Tuple, Any
import itertools

from allennlp.data.tokenizers import Token
from allennlp.data.fields import (Field, TextField, LabelField, ListField)

from src.data.dataset_readers.answer_field_generators.answer_field_generator import AnswerFieldGenerator
from src.data.dataset_readers.drop.drop_utils import SPAN_ANSWER_TYPES # TODO: Remember to remove when not required
from src.data.fields.labels_field import LabelsField
from src.data.dataset_readers.utils import find_valid_spans

@AnswerFieldGenerator.register('tagged_answer_generator')
class TaggedAnswerGenerator(AnswerFieldGenerator):
    def __init__(self,
                 ignore_question: bool,
                 flexibility_threshold: int = 1000,
                 labels: Dict[str, int] = {
                     'O': 0,
                     'B': 1,
                     'I': 2
                 }) -> None:
        super().__init__()
        self._ignore_question = ignore_question
        self._flexibility_threshold = flexibility_threshold
        self._labels = labels
        self._labels_scheme = ''.join(sorted(labels.keys()))
        if self._labels_scheme == 'BILOU':
            self._labels_scheme = 'BIOUL'

    def get_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Tuple[Dict[str, Field], bool]:
        seq_tokens: List[Token] = kwargs['seq_tokens']
        seq_wordpieces: int = kwargs['seq_wordpieces']
        question_text_index_to_token_index: List[int] = kwargs['question_text_index_to_token_index']
        question_text: str = kwargs['question_text']
        passage_text_index_to_token_index: List[int] = kwargs['passage_text_index_to_token_index']
        passage_text: str = kwargs['passage_text']
        answer_texts: List[str] = kwargs['answer_texts']
        gold_indexes: Dict[List[int]] = (kwargs['gold_indexes'] if 'gold_indexes' in kwargs 
                                          else {'question': None, 'passage': None})

        fields: Dict[str, Field] = {}

        spans_dict = {}
        all_spans = []
        is_missing_answer = False
        for i, answer_text in enumerate(answer_texts):
            answer_spans = []
            if not self._ignore_question:
                answer_spans += find_valid_spans(question_text, [answer_text], 
                                                question_text_index_to_token_index, 
                                                seq_tokens, seq_wordpieces, 
                                                gold_indexes['question'])
            answer_spans += find_valid_spans(passage_text, [answer_text], 
                                             passage_text_index_to_token_index, 
                                             seq_tokens, seq_wordpieces, 
                                             gold_indexes['passage'])
            if len(answer_spans) == 0:
                is_missing_answer = True
                continue
            spans_dict[answer_text] = answer_spans
            all_spans.extend(answer_spans)

        old_reader_behavior = kwargs['old_reader_behavior']
        if old_reader_behavior:
            answer_type = kwargs['answer_type']
            is_training = kwargs['is_training']
            if is_training:
                if answer_type in SPAN_ANSWER_TYPES:
                    if is_missing_answer:
                        all_spans = []

        if len(all_spans) > 0:
            has_answer = True

            fields['wordpiece_indices'] = self._get_wordpiece_indices_field(seq_wordpieces)

            no_answer_bios = self._get_empty_answer(seq_tokens)

            text_to_disjoint_bios: List[ListField] = []
            flexibility_count = 1
            for answer_text in answer_texts:
                spans = spans_dict[answer_text] if answer_text in spans_dict else []
                if len(spans) == 0:
                    continue

                disjoint_bios: List[LabelsField] = []
                for span_ind, span in enumerate(spans):
                    bios = self._create_sequence_labels([span], len(seq_tokens))
                    disjoint_bios.append(LabelsField(bios))

                text_to_disjoint_bios.append(ListField(disjoint_bios))
                flexibility_count *= ((2**len(spans)) - 1)

            fields['answer_as_text_to_disjoint_bios'] = ListField(text_to_disjoint_bios)

            if (flexibility_count < self._flexibility_threshold):
                # generate all non-empty span combinations per each text
                spans_combinations_dict = {}
                for key, spans in spans_dict.items():
                    spans_combinations_dict[key] = all_combinations = []
                    for i in range(1, len(spans) + 1):
                        all_combinations += list(itertools.combinations(spans, i))

                # calculate product between all the combinations per each text
                packed_gold_spans_list = itertools.product(*list(spans_combinations_dict.values()))
                bios_list: List[LabelsField] = []
                for packed_gold_spans in packed_gold_spans_list:
                    gold_spans = [s for sublist in packed_gold_spans for s in sublist]
                    bios = self._create_sequence_labels(gold_spans, len(seq_tokens))
                    bios_list.append(LabelsField(bios))

                fields['answer_as_list_of_bios'] = ListField(bios_list)
                fields['answer_as_text_to_disjoint_bios'] = ListField([ListField([no_answer_bios])])
            else:
                fields['answer_as_list_of_bios'] = ListField([no_answer_bios])

            bio_labels = self._create_sequence_labels(all_spans, len(seq_tokens))
            fields['span_bio_labels'] = LabelsField(bio_labels)

            fields['is_bio_mask'] = LabelField(1, skip_indexing=True)
        else:
            has_answer = False
            fields.update(self.get_empty_answer_fields(**kwargs))

        return fields, has_answer

    def get_empty_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Dict[str, Field]:
        seq_tokens: List[Token] = kwargs['seq_tokens']
        seq_wordpieces: int = kwargs['seq_wordpieces']

        fields: Dict[str, Field] = {}

        fields['wordpiece_indices'] = self._get_wordpiece_indices_field(seq_wordpieces)

        no_answer_bios = self._get_empty_answer(seq_tokens)

        fields['answer_as_text_to_disjoint_bios'] = ListField([ListField([no_answer_bios])])
        fields['answer_as_list_of_bios'] = ListField([no_answer_bios])
        fields['span_bio_labels'] = no_answer_bios
        fields['is_bio_mask'] = LabelField(0, skip_indexing=True)

        return fields

    @staticmethod
    def _get_empty_answer(seq_tokens: List[Token]):
        return LabelsField([0] * len(seq_tokens))

    def _create_sequence_labels(self, spans: List[Tuple[int, int]], n_labels: int):
        labels = self._labels
        labels_scheme = self._labels_scheme
        # initialize all labels to O
        labeling = [labels['O']] * n_labels

        for start, end in spans:
            if labels_scheme == 'BIO':
                # create B labels
                labeling[start] = labels['B']
                # create I labels
                labeling[start + 1 : end + 1] = [labels['I']] * (end - start)
            elif labels_scheme == 'IO':
                # create I labels
                labeling[start : end + 1] = [labels['I']] * (end - start + 1)
            elif labels_scheme == 'BIOUL':
                if end - start == 0:
                    labeling[start] = labels['U']
                else:
                    labeling[start] = labels['B']
                    labeling[start + 1 : end] = [labels['I']] * (end - start - 1)
                    labeling[end] = labels['L']
            else:
                raise Exception("Illegal labeling scheme")

        return labeling

    @staticmethod
    def _get_wordpiece_indices_field(wordpieces: List[List[int]]):
        wordpiece_token_indices = []
        ingested_indices = []
        i = 0
        while i < len(wordpieces):
            current_wordpieces = wordpieces[i]
            if len(current_wordpieces) > 1:
                wordpiece_token_indices.append(LabelsField(current_wordpieces, padding_value=-1))
                i = current_wordpieces[-1] + 1
            else:
                i += 1

        # hack to guarantee minimal length of padded number
        # according to dataset_readers.reading_comprehension.drop from allennlp)
        wordpiece_token_indices.append(LabelsField([-1], padding_value=-1))

        return ListField(wordpiece_token_indices)
