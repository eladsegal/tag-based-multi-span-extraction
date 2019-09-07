from src.preprocessing.data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.tokenizers import Token
import string
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any
import re
from allennlp.data.dataset_readers.reading_comprehension.util import (IGNORED_TOKENS,
                                                                      STRIPPED_CHARACTERS)
from src.preprocessing.data_cleaning.util import find_valid_spans

class RemoveSpansBase(CleaningObejective):
    '''
    Base class for objectives that removes spans from multi span questions.
    '''

    question_prefixes = []

    def is_fitting_objective(self, passage, question, answer):
        lowered_question = question.lower()
        return len(answer['spans']) > 1 and any(lowered_question.startswith(prefix) for prefix in self.question_prefixes)

    def clean(self, passage, question, answer, passage_tagging, question_tagging):
        passage_tokens = [Token(w) for w in passage_tagging['words']]
        spans = find_valid_spans(passage_tokens, answer['spans'])

        new_answer_texts = []

        cleaned = False

        for answer_text in answer['spans']:
            if self.should_remove_answer(answer_text):
                continue
            
            valid = True

            for span in spans:
                span_text = ' '.join(passage_tagging['words'][span[0]:span[1]+1]).lower()
                span_text = span_text.replace(' - ', '-')

                if answer_text.lower() != span_text:
                    continue
                
                if self.should_remove_span(passage_tagging['tags'][span[0]:span[1]+1]):
                    valid = False
                    cleaned = True
                    break

            if valid:
                new_answer_texts.append(answer_text)

        if not cleaned:
            return None
        
        new_answer = answer.copy()
        new_answer['spans'] = new_answer_texts

        return {'answer': new_answer}                

    def should_remove_span(self, span_tags):
        return False

    def should_remove_answer(self, answer_text):
        return False
