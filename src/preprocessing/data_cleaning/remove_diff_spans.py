from src.preprocessing.data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.tokenizers import Token
from src.preprocessing.data_cleaning.util import find_valid_spans

# *****************************************
# DID NOT SUBMIT THE DATASET FIXES FOR THIS BECAUSE THE HEURISTIC IS TOO NOISY
# *****************************************
class RemoveDiffSpans(CleaningObejective):
    '''
    Remove unrelated words from a span. All words in a span should be tagged with the same tag.
    The heuristic here is to ensure that a span is either all O or all not O to be less sensitive to tagging errors.
    '''
    
    name = "RemoveDiffSpans"

    question_prefixes = []

    whitelist = [
        # Questions allow answers should be with the same tag
        "5204020e-d2aa-4d2f-b504-784e47b50ac1",
        "ebd8398a-4559-4b90-b736-72c62edec459",
        "76604533-c5cf-4e4d-b7a2-c12e0973b4f3",
        "baf629cb-b938-4fbb-aaa2-c6d318195f53",
        "afc0a624-6091-4d89-a0d1-5035de5df2e0",
        "1d7ce8e8-c2ac-4a4a-a27d-eda8b1dc2c8c",

        # Heurisitc of taking the majority doesn't work
        "1d7ce8e8-c2ac-4a4a-a27d-eda8b1dc2c8c",

        # tagging issue
        "4616ff87-0d62-4dbc-92ba-5b05b210216b",
        "2b612783-4626-4d9a-9b4e-cc81d6f522ee",
        "6279d474-ae67-4675-8f02-ce9397fdfaf4",        
        ]

    def is_fitting_objective(self, passage, question, answer):
        return len(answer['spans']) > 2

    def clean(self, passage, question, answer, passage_tagging, question_tagging):
        passage_tokens = [Token(w) for w in passage_tagging['words']]
        spans = find_valid_spans(passage_tokens, answer['spans'])

        o_answer_texts = []
        non_o_answer_texts = []

        for answer_text in answer['spans']:
            for span in spans:
                span_text = ' '.join(passage_tagging['words'][span[0]:span[1]+1]).lower()
                span_text = span_text.replace(' - ', '-')

                if answer_text.lower() != span_text:
                    continue
                
                span_tags = passage_tagging['tags'][span[0]:span[1]+1]              

                count_non_o = sum(tag != 'O' for tag in span_tags)

                if count_non_o > 0:
                    non_o_answer_texts.append(answer_text)
                else:
                    o_answer_texts.append(answer_text)
                    
                break
       
        orig_spans_count = len(answer['spans'])

        if len(o_answer_texts) == orig_spans_count or len(non_o_answer_texts) == orig_spans_count:
            return None

        new_answer = answer.copy()
        new_answer['spans'] = non_o_answer_texts if len(non_o_answer_texts) >= len(o_answer_texts) else o_answer_texts

        return {'answer': new_answer}                
