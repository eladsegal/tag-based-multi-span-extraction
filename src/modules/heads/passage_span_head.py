from typing import Dict, Any, Union

import torch

from allennlp.nn.util import replace_masked_values, logsumexp, masked_log_softmax
from allennlp.modules import FeedForward

from src.modules.heads.head import Head
from src.modules.heads.single_span_head import SingleSpanHead

@Head.register('passage_span_head')
class PassageSpanHead(SingleSpanHead):

    def get_input_and_mask(self, kwargs: Dict[str, Any]) -> torch.LongTensor:
        return kwargs['token_representations'], kwargs['passage_mask']

    def get_gold_answer_representations(self, gold_answer_representations: Dict[str, torch.LongTensor]) -> torch.LongTensor:
        return gold_answer_representations['answer_as_passage_spans']

    def get_context(self) -> str:
        return 'p'
