import numpy as np
from typing import Dict, List, Tuple, Any
import itertools

from allennlp.data.tokenizers import Token
from allennlp.data.fields import (Field, IndexField, LabelField, ListField,
                                  ArrayField)

from src.data.dataset_readers.answer_field_generators.answer_field_generator import AnswerFieldGenerator
from src.data.dataset_readers.drop.drop_utils import get_target_numbers
from src.data.fields.labels_field import LabelsField

@AnswerFieldGenerator.register('arithmetic_answer_generator')
class ArithmeticAnswerGenerator(AnswerFieldGenerator):
    def __init__(self,
                 max_numbers_expression: int = 2,
                 special_numbers: List[float] = []) -> None:
        super().__init__()
        self._max_numbers_expression = max_numbers_expression
        self._special_numbers = special_numbers

    def get_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Tuple[Dict[str, Field], bool]:
        number_occurrences_in_passage: List[Dict[str, Any]] = kwargs['number_occurrences_in_passage']
        answer_texts: List[str] = kwargs['answer_texts']
        
        fields: Dict[str, Field] = {}

        target_numbers = get_target_numbers(answer_texts)
        
        # Get possible ways to arrive at target numbers with add/sub
        valid_expressions: List[List[int]] = \
            self._find_valid_add_sub_expressions_with_rounding(
                self._special_numbers + [number_occurrence['value'] for number_occurrence in number_occurrences_in_passage],
                target_numbers,
                self._max_numbers_expression)

        if len(valid_expressions) > 0:
            has_answer = True

            add_sub_signs_field: List[Field] = []
            special_signs_field: List[Field] = []

            for signs_for_one_add_sub_expressions in valid_expressions:
                special_signs = signs_for_one_add_sub_expressions[:len(self._special_numbers)]
                normal_signs = signs_for_one_add_sub_expressions[len(self._special_numbers):]
                add_sub_signs_field.append(LabelsField(normal_signs))
                special_signs_field.append(LabelsField(special_signs))

            fields['answer_as_expressions'] = ListField(add_sub_signs_field)
            if self._special_numbers:
                fields['answer_as_expressions_extra'] = ListField(special_signs_field)
        else:
            has_answer = False
            fields.update(self.get_empty_answer_fields(**kwargs))

        return fields, has_answer

    def get_empty_answer_fields(self,
                **kwargs: Dict[str, Any]) -> Dict[str, Field]:
        number_occurrences_in_passage: List[Dict[str, Any]] = kwargs['number_occurrences_in_passage']

        fields: Dict[str, Field] = {}

        fields['answer_as_expressions'] = ListField([LabelsField([0] * len(number_occurrences_in_passage))])
        if self._special_numbers:
            fields['answer_as_expressions_extra'] = ListField([LabelsField([0] * len(self._special_numbers))])
        
        return fields

    @staticmethod
    def _find_valid_add_sub_expressions_with_rounding(
            numbers: List[int],
            targets: List[int],
            max_number_of_numbers_to_consider: int = 2) -> List[List[int]]:
        valid_signs_for_add_sub_expressions = []
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

