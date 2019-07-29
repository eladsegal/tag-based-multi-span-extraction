from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, IndexField, LabelField, ListField, \
                                 MetadataField, SequenceLabelField, SpanField, ArrayField

import json
from overrides import overrides
from typing import Dict, List, Union, Tuple, Any
import numpy as np

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
        self.answer_types = answer_types
        self.max_instances = max_instances

    @overrides
    def _read(self, file_path):
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        instances = []
        for passage_id, passage_info in dataset.items():
            passage_text = passage_info['passage']
            passage_tokens = self.tokenizer.tokenize(passage_text)

            #Note: NAQANET addded split by hyphen here

            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []

                answer_type = get_answer_type(question_answer['answer'])

                if "answer" in question_answer:
                    if self.answer_types is not None and answer_type not in self.answer_types:
                        continue

                    answer_annotations.append(question_answer["answer"])

                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]
                                    
                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 answer_type,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 passage_tokens)
                
                if instance is not None:
                    instances.append(instance)

                if self.max_instances > 0 and self.max_instances <= len(instances):
                    return instances

        return instances

    @overrides
    def text_to_instance(self, 
                         question_text: str,
                         passage_text: str,
                         answer_type: str,
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict] = None,
                         passage_tokens: List[Token] = None) -> Union[Instance, None]:

        fields: Dict[str, Field] = {}

        question_tokens = self.tokenizer.tokenize(question_text)
        # Note: NAQANET add split by hyphen here

        question_passage_tokens = [Token('[CLS]')] + question_tokens + [Token("[SEP]")] + passage_tokens
        question_passage_tokens += [Token('[SEP]')]
        question_and_passage_field = TextField(question_passage_tokens, self.token_indexers)
        fields['question_and_passage'] = question_and_passage_field

        if answer_type == 'multiple_span':
            # We use first answer annotation, like in NABERT
            bio_labels, bio_mask = generate_bio_labels(question_passage_tokens, answer_annotations[0])
        else:
            bio_labels, bio_mask = np.zeros(len(question_and_passage_field)), np.zeros(len(question_and_passage_field))

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

def get_answer_type(answers):
    if answers['number']:
        return 'number'
    elif answers['spans']:
        if len(answers['spans']) == 1:
            return 'single_span'
        return 'multiple_span'
    elif any(answers['date'].values()):
        return 'date'

# A mock for a method that creates an array of b(0),i(1),o(2) labels for the tokens
def generate_bio_labels(question_passage_tokens, answer_annotation):
    return np.ones(len(question_passage_tokens)), np.ones(len(question_passage_tokens))