from collections import defaultdict
import json
from overrides import overrides
from typing import Dict, List, Tuple, Optional

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, MetadataField, SequenceLabelField, ArrayField
import numpy as np

from src.preprocessing.utils import SPAN_ANSWER_TYPES
from src.preprocessing.utils import get_answer_type, find_span


@DatasetReader.register("multi_span_drop")
class BertDropReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy=False,
                 answer_types: List[str] = None,
                 max_pieces: int = 512,
                 max_instances: int = -1):
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.answer_types = answer_types or ['single_span', 'multiple_span', 'number', 'date']
        self.max_pieces = max_pieces
        self.max_instances = max_instances

    @overrides
    def _read(self, file_path):
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        instances = []
        for passage_id, passage_data in dataset.items():
            passage_text = passage_data['passage']
            passage_tokens = self.tokenizer.tokenize(passage_text)

            # NAQANET added split by hyphens here

            for qa_pair in passage_data["qa_pairs"]:
                question_id = qa_pair["query_id"]
                question_text = qa_pair["question"].strip()
                answer = qa_pair['answer']

                answer_type = get_answer_type(answer)
                if answer_type not in self.answer_types:
                    continue

                answer_annotations: List[Dict] = list()
                answer_annotations.append(answer)

                if "validated_answers" in qa_pair:
                    answer_annotations += qa_pair["validated_answers"]
                                    
                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 answer_type,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 passage_tokens)
                if instance is not None:
                    instances.append(instance)

                if 0 < self.max_instances <= len(instances):
                    break

        return instances

    @overrides
    def text_to_instance(self, 
                         question_text: str,
                         passage_text: str,
                         answer_type: str,
                         question_id: str,
                         passage_id: str,
                         answer_annotations: List[Dict],
                         passage_tokens: List[Token] = None) -> Optional[Instance]:

        fields: Dict[str, Field] = {}

        # tokenize question
        question_tokens = self.tokenizer.tokenize(question_text)
        # Note: NAQANET add split by hyphen here

        # truncate question+passage
        qp_tokens: List[Token] = []
        qp_tokens.extend([Token('[CLS]')] + question_tokens + [Token('[SEP]')] + passage_tokens)
        qp_tokens = qp_tokens[:self.max_pieces-1]  # `-1` as we still need to add [SEP] at the end
        qp_tokens.append(Token('[SEP]'))

        # create question+passage token field
        qp_field = TextField(qp_tokens, self.token_indexers)
        fields['question_and_passage'] = qp_field

        # handle span questions
        if answer_type in SPAN_ANSWER_TYPES:
            qp_token_indices: Dict[Token, List[int]] = defaultdict(list)
            for i, token in enumerate(qp_tokens):
                qp_token_indices[token].append(i)

            # We use the first answer annotation, like in NABERT
            answer_texts = answer_annotations[0]['spans']

            spans = []
            for answer_text in answer_texts:
                answer_tokens = self.tokenizer.tokenize(answer_text)
                answer_span = find_span(answer_tokens, qp_token_indices, len(qp_field))
                if len(answer_span) != 1:
                    return None
                spans.extend(answer_span)

            bio_labels = create_bio_labels(spans, len(qp_field))
            fields['span_labels'] = SequenceLabelField(bio_labels, sequence_field=qp_field)

            # in a word broken up into pieces, every piece except the first should be ignored when calculating the loss
            wordpiece_mask = [not token.text.startswith('##') for token in qp_tokens]
            wordpiece_mask = np.array(wordpiece_mask)
            fields['span_wordpiece_mask'] = ArrayField(wordpiece_mask)

        metadata = {
            'original_passage': passage_text,
            "original_question": question_text,
            "passage_id": passage_id,
            "question_id": question_id,
            "answer_annotations": answer_annotations,
            'question_tokens': [token.text for token in question_tokens],
            'passage_tokens': [token.text for token in passage_tokens],
            "answer_type": answer_type
          }

        fields['metadata'] = MetadataField(metadata)
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
