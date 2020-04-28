from typing import Dict, List, Tuple, Any

from allennlp.data.fields import Field, LabelField, ListField
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader

from src.data.dataset_readers.answer_field_generators.answer_field_generator import AnswerFieldGenerator
from src.data.dataset_readers.drop.drop_utils import get_target_numbers

@AnswerFieldGenerator.register('count_answer_generator')
class CountAnswerGenerator(AnswerFieldGenerator):
    def __init__(self,
                 max_count: int = 10) -> None:
        super().__init__()
        self._max_count = max_count

    def get_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Tuple[Dict[str, Field], bool]:
        answer_texts: List[str] = kwargs['answer_texts']
        
        fields: Dict[str, Field] = {}

        target_numbers = get_target_numbers(answer_texts)
        
        numbers_for_count = list(range(self._max_count + 1))
        valid_counts: List[int] = DropReader.find_valid_counts(numbers_for_count, target_numbers)

        if len(valid_counts) > 0:
            has_answer = True

            counts_field: List[Field] = [LabelField(count_label, skip_indexing=True) for count_label in valid_counts]
            fields['answer_as_counts'] = ListField(counts_field)
        else:
            has_answer = False
            fields.update(self.get_empty_answer_fields(**kwargs))

        return fields, has_answer

    def get_empty_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Dict[str, Field]:
        answer_texts: List[str] = kwargs['answer_texts']

        fields: Dict[str, Field] = {}

        counts_field: List[Field] = [LabelField(-1, skip_indexing=True)]
        fields['answer_as_counts'] = ListField(counts_field)

        return fields
