from typing import Dict, Any, Union

import torch

from allennlp.nn.util import replace_masked_values, logsumexp, masked_log_softmax
from allennlp.modules import FeedForward

from src.modules.heads.head import Head
from src.modules.heads.single_span_head import SingleSpanHead

@Head.register('question_span_head')
class QuestionSpanHead(SingleSpanHead):

    def get_input_and_mask(self, kwargs: Dict[str, Any]) -> torch.LongTensor:
        token_representations = kwargs['token_representations']
        passage_vector = kwargs['passage_summary_vector']
        question_mask = kwargs['question_mask']
        question_passage_special_tokens_mask = kwargs['question_passage_special_tokens_mask']

        num_of_tokens =  question_mask.sum(-1).max() + question_passage_special_tokens_mask.sum(-1).max()
        question_tokens = token_representations[:,:num_of_tokens]
        question_mask = question_mask[:,:num_of_tokens]

        return torch.cat([question_tokens, passage_vector.unsqueeze(1).repeat(1, question_tokens.size(1), 1)], -1), question_mask

    def get_gold_answer_representations(self, gold_answer_representations: Dict[str, torch.LongTensor]) -> torch.LongTensor:
        return gold_answer_representations['answer_as_question_spans']

    def get_context(self) -> str:
        return 'q'
