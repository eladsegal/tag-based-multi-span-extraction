from enum import Enum

class AnswerType(Enum):
    SINGLE_SPAN = 'single_span'
    MULTI_SPAN = 'multiple_span'

ALL_ANSWER_TYPES = [AnswerType.SINGLE_SPAN.value, AnswerType.MULTI_SPAN.value]

def get_answer_type(answer):
    if answer:
        if len(answer) == 1:
            return AnswerType.SINGLE_SPAN.value
        else:
            return AnswerType.MULTI_SPAN.value
    else:
        return None

def standardize_dataset(dataset, standardize_text):
    for article in dataset['data']:
        for passage_info in article['paragraphs']:
            passage_info['context'] = standardize_text(passage_info['context'])

            for qa_pair in passage_info['qas']:
                qa_pair['question'] = standardize_text(qa_pair['question'])

                if 'answer' in qa_pair:
                    answer = qa_pair['answer']
                    if 'spans' in answer:
                        answer['spans'] = [standardize_text(span) for span in answer['spans']]

                if 'answers' in qa_pair:
                    for answer in qa_pair['answers']:
                        answer['text'] = standardize_text(answer['text'])
    return dataset