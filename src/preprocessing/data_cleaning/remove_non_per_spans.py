from src.preprocessing.data_cleaning.cleaning_objective import CleaningObejective
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token
from src.preprocessing.data_cleaning.remove_spans_base import RemoveSpansBase

class RemoveNonPerSpans(RemoveSpansBase):
    '''
    Remove spans that are classified as O for multi span questions where we don't expect O
    '''

    name = "RemoveNonPerSpans"

    question_prefixes = [
        "who scored",
        "who caught",
        "who kicked",
        "which players",
        "what players",
    ]
    
    whitelist = [
            # Two questions where the result is ORG, ignoring them and removing the ORG for the PER expected to get better results.
            "370330a3-3c6a-47d0-ab82-abe55d6bc6d3",
            "26739a0f-3e6e-49ee-9595-add767692a82",

            # Person is no recognized as PER
            "afa06d9f-51fd-4488-a676-ea4c6ac05010",
            "0f801133-dfd0-42fd-b5d1-789a1a0cc735",
            "02ba8a32-bf89-4d0d-bc52-61f3d4ca96c7",
        ]

    def should_remove_span(self, span_tags):
        return all(not tag.endswith('PER') for tag in span_tags)

    def should_remove_answer(self, answer_text):
        return answer_text.isdigit()
