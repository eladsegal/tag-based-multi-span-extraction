from typing import Dict, List, Tuple, Any

from allennlp.data.fields import Field, ListField, SpanField
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader

from src.data.dataset_readers.answer_field_generators.answer_field_generator import AnswerFieldGenerator
from src.data.dataset_readers.drop.drop_utils import SPAN_ANSWER_TYPES # TODO: Remember to remove when not required
from src.data.fields.labels_field import LabelsField
from src.data.dataset_readers.utils import find_valid_spans

@AnswerFieldGenerator.register('span_answer_generator')
class SpanAnswerGenerator(AnswerFieldGenerator):
    def __init__(self, text_type: str) -> None:
        super().__init__()
        self._text_type = text_type # passage / question

    def get_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Tuple[Dict[str, Field], bool]:
        seq_tokens: List[Token] = kwargs['seq_tokens']
        seq_field: List[Token] = kwargs['seq_field']
        seq_wordpieces: int = kwargs['seq_wordpieces']
        text_index_to_token_index: List[int] = kwargs[f'{self._text_type}_text_index_to_token_index']
        text: str = kwargs[f'{self._text_type}_text']
        answer_texts: List[str] = kwargs['answer_texts']
        gold_indexes: Dict[List[int]] = (kwargs['gold_indexes'] if 'gold_indexes' in kwargs 
                                          else {'question': None, 'passage': None})

        fields: Dict[str, Field] = {}

        spans_dict = {}
        all_spans = []
        is_missing_answer = False
        for answer_text in answer_texts:
            answer_spans = find_valid_spans(text, [answer_text], 
                                            text_index_to_token_index, 
                                            seq_tokens, seq_wordpieces, 
                                            gold_indexes[self._text_type])
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

            span_fields: List[Field] = [SpanField(span[0], span[1], seq_field) for span in all_spans]
            fields[f'answer_as_{self._text_type}_spans'] = ListField(span_fields)
        else:
            has_answer = False
            fields.update(self.get_empty_answer_fields(**kwargs))

        return fields, has_answer

    def get_empty_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Dict[str, Field]:
        seq_field: List[str] = kwargs['seq_field']

        fields: Dict[str, Field] = {}

        span_fields: List[Field] = [SpanField(-1, -1, seq_field)]        
        fields[f'answer_as_{self._text_type}_spans'] = ListField(span_fields)

        return fields
