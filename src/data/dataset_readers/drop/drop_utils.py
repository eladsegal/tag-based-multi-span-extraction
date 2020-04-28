from typing import Dict, List, Tuple, Any

from enum import Enum
import re
import string

from word2number.w2n import word_to_num

from src.data.fields.labels_field import LabelsField
from allennlp.data.fields import ListField

class AnswerType(Enum):
    SINGLE_SPAN = 'single_span'
    MULTI_SPAN = 'multiple_span'
    NUMBER = 'number'
    DATE = 'date'

class AnswerAccessor(Enum):
    SPAN = 'spans'
    NUMBER = 'number'
    DATE = 'date'

SPAN_ANSWER_TYPES = [AnswerType.SINGLE_SPAN.value, AnswerType.MULTI_SPAN.value]
ALL_ANSWER_TYPES = SPAN_ANSWER_TYPES + [AnswerType.NUMBER.value, AnswerType.DATE.value]

def get_answer_type(answer):
    if answer['number']:
        return AnswerType.NUMBER.value
    elif answer['spans']:
        if len(answer['spans']) == 1:
            return AnswerType.SINGLE_SPAN.value
        return AnswerType.MULTI_SPAN.value
    elif any(answer['date'].values()):
        return AnswerType.DATE.value
    else:
        return None

def get_number_from_word(word):
    punctuation = string.punctuation.replace('-', '')
    word = word.strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if re.match('^\d*1st$', word):  # ending in '1st'
                    number = int(word[:-2])
                elif re.match('^\d*2nd$', word):  # ending in '2nd'
                    number = int(word[:-2])
                elif re.match('^\d*3rd$', word):  # ending in '3rd'
                    number = int(word[:-2])
                elif re.match('^\d+th$', word):  # ending in <digits>th
                    # Many occurrences are when referring to centuries (e.g "the *19th* century")
                    number = int(word[:-2])
                elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                    # Decades, e.g. "1960s".
                    # Other sequences of digits ending with s (there are 39 of these in the training
                    # set), do not seem to be arithmetically related, as they are usually proper
                    # names, like model numbers.
                    number = int(word[:-1])
                elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                    # per square kilometer, e.g "73/km²" or "3057.4/km2"
                    if '.' in word:
                        number = float(word[:-4])
                    else:
                        number = int(word[:-4])
                elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                    # per month, e.g "1050.95/month"
                    if '.' in word:
                        number = float(word[:-6])
                    else:
                        number = int(word[:-6])
                else:
                    return None
    return number

def extract_number_occurrences(number_extraction_tokens, alignment):
    number_occurrences = []
    for i, token in enumerate(number_extraction_tokens):
        number = get_number_from_word(token.text)

        if number is not None:
            number_occurrences.append({
                'value': number,
                'indices': alignment[i]
            })
    return number_occurrences

def clipped_passage_num(number_occurrences_in_passage, clipped_length):
    if not number_occurrences_in_passage or number_occurrences_in_passage[-1]['indices'][0] < clipped_length:
        return number_occurrences_in_passage
    lo = 0
    hi = len(number_occurrences_in_passage) - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if number_occurrences_in_passage[mid]['indices'][0] < clipped_length:
            lo = mid + 1
        else:
            hi = mid
    
    last_number_occurrence = number_occurrences_in_passage[lo - 1]
    last_number_occurrence['indices'] = [index for index in last_number_occurrence['indices'] if index < clipped_length]

    return number_occurrences_in_passage[:lo]

def get_target_numbers(answer_texts):
    target_numbers = []
    for answer_text in answer_texts:
        number = get_number_from_word(answer_text)
        if number is not None:
            target_numbers.append(number)
    return target_numbers

def standardize_dataset(dataset, standardize_text):
    for passage_info in dataset.values():
        passage_info['original_passage'] = passage_info['passage']
        passage_info['passage'] = standardize_text(passage_info['passage'])
        for qa_pair in passage_info["qa_pairs"]:
            qa_pair['question'] = standardize_text(qa_pair['question'])

            if 'answer' in qa_pair:
                answer = qa_pair['answer']
                if 'spans' in answer:
                    answer['spans'] = [standardize_text(span) for span in answer['spans']]
            if 'validated_answers' in qa_pair:
                for validated_answer in qa_pair['validated_answers']:
                    if 'spans' in answer:
                        validated_answer['spans'] = [standardize_text(span) for span in validated_answer['spans']]
    return dataset

def standardize_dataset_new(dataset, standardize_text):
    # The idea is to save the original fields.
    # The standardized fields will be used for everything except for the evaluation.
    # That also means that a predicted span should be mapped to its equivalent in the original field.
    for passage_info in dataset.values():
        passage_info['passage'] = standardize_text(passage_info['passage']).strip()
        for qa_pair in passage_info['qa_pairs']:
            qa_pair['question'] = standardize_text(qa_pair['question']).strip()

            answer = qa_pair['answer']
            if 'answer' in qa_pair:
                if 'spans' in answer:
                    answer['standardized_spans'] = [standardize_text(span).strip() for span in answer['spans']]
            if 'validated_answers' in qa_pair:
                for validated_answer in qa_pair['validated_answers']:
                    if 'spans' in answer:
                        validated_answer['standardized_spans'] = [standardize_text(span).strip() for span in validated_answer['spans']]
    return dataset

def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
    # From allennlp's DropReader.
    # Changed in order to use the standardized span answers

    answer_type = None
    if "standardized_spans" in answer_annotation and answer_annotation["standardized_spans"]:
        answer_type = "standardized_spans"
    elif answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts: List[str] = []
    if answer_type is None:  # No answer
        pass
    elif answer_type == "standardized_spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
        answer_type = "spans"
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [answer_content[key]
                        for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts

def get_number_indices_field(number_occurrences_in_passage: List[Dict[str, Any]]):
    number_token_indices = \
        [LabelsField(number_occurrence['indices'], padding_value=-1) for number_occurrence in number_occurrences_in_passage]

    return ListField(number_token_indices)
