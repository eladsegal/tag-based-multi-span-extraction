import json
import re
from typing import List, Tuple

from src.preprocessing.utils import (SPAN_ANSWER_TYPE, SPAN_ANSWER_TYPES, NUMBER_ANSWER_TYPE,
                                     DATE_ANSWER_TYPE)
from src.preprocessing.utils import get_answer_type, deep_dict_update

MISSING_SPANS = 'missing_spans'
AMBIGUOUS_SPANS = 'ambiguous_spans'
INVALID_TYPE = 'invalid_type'
ERRORS_KEYNAME = 'errors'
ERROR_TYPES = {MISSING_SPANS, AMBIGUOUS_SPANS, INVALID_TYPE}


class DatasetAnnotator:
    def __init__(self, dataset_path, annotated_dataset_output_path=None):
        self.dataset_path = dataset_path
        self.annotated_dataset_output_path = \
            annotated_dataset_output_path or dataset_path.replace('.json', '_annotated.json')

    def _load_dataset(self):
        with open(self.dataset_path) as dataset_file:
            return json.load(dataset_file)

    def _save_annotated_dataset(self, annotated_dataset):
        with open(self.annotated_dataset_output_path, 'w') as f:
            json.dump(annotated_dataset, f, indent=2)


class DatasetErrorAnnotator(DatasetAnnotator):

    def annotate_errors(self):
        self.annotate_span_errors()

        original_dataset_path = self.dataset_path
        self.dataset_path = self.annotated_dataset_output_path

        self.annotate_answer_type_errors()
        self.dataset_path = original_dataset_path

    def annotate_span_errors(self):
        dataset = self._load_dataset()

        for passage_data in dataset.values():
            passage_text = passage_data['passage']
            for qa_pair in passage_data['qa_pairs']:
                question_text = qa_pair['question']
                answer = qa_pair['answer']

                # skip if not a span question
                answer_type = get_answer_type(answer)
                if answer_type not in SPAN_ANSWER_TYPES:
                    continue
                else:
                    answer_texts = answer['spans']

                answer_indices_with_missing_span = []
                answer_indices_with_ambiguous_span = []
                all_q_spans: List[List[Tuple]] = []
                all_p_spans: List[List[Tuple]] = []
                for answer_idx, answer_text in enumerate(answer_texts):
                    q_spans = [m.span() for m in re.finditer(re.escape(answer_text), question_text)]
                    p_spans = [m.span() for m in re.finditer(re.escape(answer_text), passage_text)]

                    # missing span
                    if not q_spans and not p_spans:
                        answer_indices_with_missing_span.append(answer_idx)

                    # ambiguous span
                    if len(q_spans) + len(p_spans) > 1:
                        answer_indices_with_ambiguous_span.append(answer_idx)
                        all_q_spans.append(q_spans)
                        all_p_spans.append(p_spans)

                if answer_indices_with_missing_span:
                    missing_spans = self._create_missing_spans_error(answer_indices_with_missing_span)
                    deep_dict_update(answer, missing_spans)

                if answer_indices_with_ambiguous_span:
                    ambiguous_spans = self._create_ambiguous_spans_error(
                        answer_indices_with_ambiguous_span, all_q_spans, all_p_spans)
                    deep_dict_update(answer, ambiguous_spans)

        self._save_annotated_dataset(dataset)

    def annotate_answer_type_errors(self):
        dataset = self._load_dataset()

        for passage_data in dataset.values():
            for qa_pair in passage_data['qa_pairs']:
                answer = qa_pair['answer']

                # get all answer types
                answer_types = []
                if answer['spans']:
                    answer_types.append(SPAN_ANSWER_TYPE)
                if answer['number']:
                    answer_types.append(NUMBER_ANSWER_TYPE)
                if any(answer['date'].values()):
                    answer_types.append(DATE_ANSWER_TYPE)

                if len(answer_types) != 1:
                    type_error = self._create_answer_type_error(answer_types)
                    deep_dict_update(answer, type_error)

        self._save_annotated_dataset(dataset)

    @staticmethod
    def _create_missing_spans_error(answer_indices):
        return {
            ERRORS_KEYNAME: {
                MISSING_SPANS: {'answer_indices': answer_indices}}}

    @staticmethod
    def _create_ambiguous_spans_error(answer_indices, q_spans, p_spans):
        return {
            ERRORS_KEYNAME: {
                AMBIGUOUS_SPANS: {'answer_indices': answer_indices,
                                   'q_spans': q_spans,
                                   'p_spans': p_spans
                                    }}}
    @staticmethod
    def _create_answer_type_error(types):
        return {
            ERRORS_KEYNAME: {
                INVALID_TYPE: types}}
