import json

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

                # find answers with missing span
                answer_indices_with_missing_span = []
                for answer_idx, answer_text in enumerate(answer_texts):
                    if answer_text not in passage_text and answer_text not in question_text:
                        answer_indices_with_missing_span.append(answer_idx)

                # update the answer with the error
                if answer_indices_with_missing_span:
                    error = self.create_missing_spans_error(answer_indices_with_missing_span)
                    deep_dict_update(answer, error)

        # save the new annotated dataset
        with open(self.annotated_dataset_output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

    @staticmethod
    def create_missing_spans_error(answer_indices):
        return {
            'errors': {
                'missing_spans': {'answer_indices': answer_indices,
                                  'possible_correct_passage': -1}}}
