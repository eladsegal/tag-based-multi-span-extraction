from data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token

class TopTwoNumbers(CleaningObejective):
    '''
    The training dataset contain 154 questions that start with "What are the top two longest..." and 92 question that start with "What are the two shortest ..."
    The expected answer to most of them is either numeric spans or numeric with "-yard" spans.
    However, for some of them the expected answer is the data, contain some unrelated names from the passage in addition to the good answer.
    The purpose here is to remove the unrelated names.
    '''

    name = "TopTwoNumbers"

    question_prefixes = [
        "what are the top two longest",
        "what are the two shortest"
    ]

    def is_fitting_objective(self, passage, question, answer):
        lowered_question = question.lower()
        return len(answer['spans']) > 2 and any(lowered_question.startswith(prefix) for prefix in self.question_prefixes)

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



