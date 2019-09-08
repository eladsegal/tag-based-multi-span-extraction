from src.preprocessing.data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token
from src.preprocessing.data_cleaning.remove_spans_base import RemoveSpansBase

class RemoveNonNumberSpans(RemoveSpansBase):
    '''
    The training dataset contain 154 questions that start with "What are the top two longest..." and 92 question that start with "What are the two shortest ..."
    The expected answer to most of them is either numeric spans or numeric with "-yard" spans.
    However, for some of them the expected answer is the data, contain some unrelated names from the passage in addition to the good answer.
    The purpose here is to remove the unrelated names.
    '''

    name = "RemoveNonNumberSpans"

    question_prefixes = [
        "what are the top two longest",
        "what are the two shortest",
        "what are the three shortest",
        "how long",
        "how many yards",
    ]

    whitelist = [
            "5f670475-4859-41e0-8547-9ce87d4fb17d"
            ]

    def clean(self, passage, question, answer, passage_tagging, question_tagging):
        passage_tokens = [Token(w) for w in passage_tagging['words']]
        spans = DropReader.find_valid_spans(passage_tokens, answer['spans'])

        new_answer_texts = []

        cleaned = False

        for answer_text in answer['spans']:
            valid = True

            for span in spans:
                span_text = ' '.join(passage_tagging['words'][span[0]:span[1]+1]).lower()

                if answer_text.lower() != span_text:
                    continue

                if any(tag != 'O' for tag in passage_tagging['tags'][span[0]:span[1]+1]):
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
        return any(tag != 'O' for tag in span_tags)

    def should_remove_answer(self, answer_text):
        return answer_text.split()

