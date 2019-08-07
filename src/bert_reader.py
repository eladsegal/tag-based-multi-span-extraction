from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, IndexField, LabelField, ListField, \
                                 MetadataField, SequenceLabelField, SpanField, ArrayField

from collections import defaultdict
import json
import logging
from overrides import overrides
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

@DatasetReader.register("multi_span_drop")
class BertDropReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy=False,
                 answer_types: List[str] = None,
                 max_instances: int = -1):
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.answer_types = answer_types or ['single_span', 'multiple_span', 'number', 'date']
        self.max_instances = max_instances

    @overrides
    def _read(self, file_path):
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        instances = []
        for passage_id, passage_data in dataset.items():
            passage_text = passage_data['passage']
            passage_tokens = self.tokenizer.tokenize(passage_text)

            #Note: NAQANET addded split by hyphen here

            for qa_pair in passage_data["qa_pairs"]:
                question_id = qa_pair["query_id"]
                question_text = qa_pair["question"].strip()
                answer = qa_pair['answer']

                answer_type = get_answer_type(answer)
                if answer_type not in self.answer_types:
                    if answer_type is None:
                        logger.warning(f'answer for question `{question_text}` has no valid answer type.\n'
                                       f'valid answer types: {self.answer_types}\n'
                                       f'query_id: {question_id}')
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

                if self.max_instances >= len(instances):
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

        question_tokens = self.tokenizer.tokenize(question_text)
        # Note: NAQANET add split by hyphen here

        qp_tokens: List[Token] = \
            [Token('[CLS]')] + question_tokens + [Token('[SEP]')] + passage_tokens + [Token('[SEP]')]

        qp_field = TextField(qp_tokens, self.token_indexers)
        fields['question_and_passage'] = qp_field

        if answer_type == 'single_span' or 'multiple_span':
            # We use first answer annotation, like in NABERT
            bio_labels, bio_mask = self.generate_bio_labels(qp_tokens, answer_annotations[0]['spans'])
            if bio_labels is None or bio_mask is None:
                return None

        fields['multi_span_bio'] = ArrayField(array=bio_labels)
        fields['multi_span_bio_mask'] = ArrayField(array=bio_mask)

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

    def generate_bio_labels(self, qp_tokens, answer_texts):

        qp_token_indices: Dict[Token, List[int]] = defaultdict(list)
        for i, token in enumerate(qp_tokens):
            qp_token_indices[token].append(i)

        spans = list()
        for answer_text in answer_texts:
            answer_tokens = self.tokenizer.tokenize(answer_text)
            span = find_span(answer_tokens, qp_token_indices, len(qp_tokens))

            if len(span) != 1:
                if len(span) == 0:
                    logger.warning(f'no match was found for answer `{answer_text}` in tokens: `{qp_tokens}`')
                # if len(span) > 1:
                #     logger.warning(f'multiple matches were found for answer `{answer_text}` in tokens: `{qp_tokens}`\n'
                #                    f'span indices: `{span}`')
                return None, None
            spans += span

        # create bio labels from the spans
        # create mask from the spans

        return None, None


def find_span(answer_tokens: List[Token], qp_token_indices: Dict[Token, List[int]],
              num_qp_tokens) -> List[Tuple[int, int]]:
    num_answer_tokens = len(answer_tokens)
    span = []
    for span_start in qp_token_indices[answer_tokens[0]]:
        if num_answer_tokens == 1:
            span.append((span_start, span_start))
        elif span_start + num_answer_tokens - 1 <= num_qp_tokens:
            for i, answer_token in enumerate(answer_tokens[1:], 1):
                if not span_start + i in qp_token_indices[answer_token]:
                    break
            else:
                span_end = span_start + i  # span_end is inclusive
                span.append((span_start, span_end))
    return span



def get_answer_type(answer):
    if answer['number']:
        return 'number'
    elif answer['spans']:
        if len(answer['spans']) == 1:
            return 'single_span'
        return 'multiple_span'
    elif any(answer['date'].values()):
        return 'date'
    else:
        return None
