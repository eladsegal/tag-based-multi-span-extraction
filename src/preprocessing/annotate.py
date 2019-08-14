import json
import re
from typing import List, Tuple

from src.preprocessing.utils import SPAN_ANSWER_TYPES
from src.preprocessing.utils import get_answer_type, deep_dict_update


class DatasetAnnotator:
    def __init__(self, dataset_path, annotated_dataset_output_path=None):
        self.dataset_path = dataset_path
        self.annotated_dataset_output_path = \
            annotated_dataset_output_path or dataset_path.replace('.json', '_annotated.json')


class DatasetErrorAnnotator(DatasetAnnotator):

    def annotate_missing_spans(self):
        with open(self.dataset_path) as dataset_file:
            dataset = json.load(dataset_file)

        for passage_id, passage_data in dataset.items():
            passage_text = passage_data['passage']
            for qa_pair in passage_data["qa_pairs"]:
                question_text = qa_pair["question"]
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
                    missing_spans = self.create_missing_spans_error(answer_indices_with_missing_span)
                    deep_dict_update(answer, missing_spans)

                if answer_indices_with_ambiguous_span:
                    ambiguous_spans = self.create_ambiguous_spans_error(
                        answer_indices_with_ambiguous_span, all_q_spans, all_p_spans)
                    deep_dict_update(answer, ambiguous_spans)

        # save the new annotated dataset
        with open(self.annotated_dataset_output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

    @staticmethod
    def create_missing_spans_error(answer_indices):
        return {
            'errors': {
                'missing_spans': {'answer_indices': answer_indices}}}

    @staticmethod
    def create_ambiguous_spans_error(answer_indices, q_spans, p_spans):

        return {
            'errors': {
                'ambiguous_spans': {'answer_indices': answer_indices,
                                    'q_spans': q_spans,
                                    'p_spans': p_spans
                                    }}}
