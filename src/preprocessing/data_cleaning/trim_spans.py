from src.preprocessing.data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token

# *****************************************
# DID NOT SUBMIT THE DATASET FIXES FOR THIS YET
# *****************************************
class TrimSpans(CleaningObejective):
    '''
    Remove unrelated words from a span. All words in a span should be tagged with the same tag.
    The heuristic here is to ensure that a span is either all O or all not O to be less sensitive to tagging errors.
    '''
    
    name = "TrimSpans"

    question_prefixes = [
        "which team",
        "which two teams",
        "who scored",
        "who caught",
        "who kicked",
        "which players",
        "what players",
]

    def is_fitting_objective(self, passage, question, answer):
        return len(answer['spans']) > 1 and any(len(span.split()) > 1 for span in answer['spans'])

    def clean(self, passage, question, answer, passage_tagging, question_tagging):
        passage_tokens = [Token(w) for w in passage_tagging['words']]
        spans = DropReader.find_valid_spans(passage_tokens, answer['spans'])

        if not spans:
            return None

        new_answer_texts = []

        cleaned = False

        for answer_text in answer['spans']:
            if len(answer_text.split()) <= 1:
                continue

            new_answer_text = answer_text

            for span in spans:
                span_text = ' '.join(passage_tagging['words'][span[0]:span[1]+1]).lower()

                if answer_text.lower() != span_text:
                    continue
                
                span_tags = passage_tagging['tags'][span[0]:span[1]+1]

                count_o = sum(tag == 'O' for tag in span_tags)
                other_than_o = len(span_tags) - count_o

                if count_o == 0 or other_than_o == 0:
                    break

                tags_to_trim = ['O'] if count_o <= other_than_o else ['ORG', 'LOC', 'PER', 'MISC']

                # Remove words only from the start and from the end to keep a valid span
                span_words = passage_tagging['words'][span[0]:span[1]+1]
                words_count = len(span_words)

                remove_from_start = True
                remove_from_end = True

                for i in range(words_count):
                    if remove_from_start and all(not span_tags[i].endswith(tag) for tag in tags_to_trim):
                        remove_from_start = False

                    if remove_from_end and all(not span_tags[words_count - i - 1].endswith(tag) for tag in tags_to_trim):
                        remove_from_end = False

                    if remove_from_start:
                        del span_words[0]

                    if remove_from_end:
                        del span_words[-1]

                    if not remove_from_end and not remove_from_start:
                        break

                new_answer_text = ' '.join(span_words)
                cleaned = True
                break
                    
            new_answer_texts.append(new_answer_text)
       
        if not cleaned:
            return None

        new_answer = answer.copy()
        new_answer['spans'] = new_answer_texts

        return {'answer': new_answer}                
