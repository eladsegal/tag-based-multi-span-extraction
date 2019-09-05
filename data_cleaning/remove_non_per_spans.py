from data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token
from data_cleaning.remove_spans_base import RemoveSpansBase

class RemoveNonPerSpans(RemoveSpansBase):
    '''
    Remove spans that are classified as O for multi span questions where we don't expect O
    '''

    name = "RemoveNonPerSpans"

    question_prefixes = [
        "who scored",
        "who caught",
        "who kicked",
    ]

    whitelist = [
        ]

    def should_remove_span(self, span_tags):
        return all(not tag.endswith('PER') and not tag.endswith('ORG') for tag in span_tags)

    def should_remove_answer(self, answer_text):
        return answer_text.isdigit()


    #TEST
    # 370330a3-3c6a-47d0-ab82-abe55d6bc6d3
    # 26739a0f-3e6e-49ee-9595-add767692a82