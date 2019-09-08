from src.preprocessing.data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token
from src.preprocessing.data_cleaning.remove_spans_base import RemoveSpansBase

class RemoveNonOrgLocMiscSpans(RemoveSpansBase):
    '''
    Remove spans that are classified as O for multi span questions where we don't expect O
    '''

    name = "RemoveNonOrgLocMiscSpans"

    question_prefixes = [
        "which team",
        "which two teams"
    ]
   
    whitelist = [
        # Team is not tagged as ORG nor LOC nor MISC
        "291db275-c598-4192-ab26-c3f450eea4fd",
        "64a6b892-8ffb-4e73-8738-5161fd1f1b3c",
        ]

    def should_remove_span(self, span_tags):
        # LOC improves prediction for teams
        return all(not tag.endswith('ORG') and not tag.endswith('LOC') and not tag.endswith('MISC') for tag in span_tags)

    def should_remove_answer(self, answer_text):
        return answer_text.isdigit()
