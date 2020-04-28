from typing import Dict, Any, Union

import torch

from allennlp.nn.util import replace_masked_values, logsumexp
from allennlp.modules import FeedForward

from src.modules.heads.head import Head

@Head.register('count_head')
class CountHead(Head):
    def __init__(self,
                 output_layer: FeedForward,
                 max_count: int) -> None:
        super().__init__()
        self._output_layer = output_layer
        self._max_count = max_count

    def forward(self,
                passage_summary_vector: torch.LongTensor,
                **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_count)
        logits = self._output_layer(passage_summary_vector)
        log_probs = torch.nn.functional.log_softmax(logits, -1)

        # Info about the best count number prediction
        # Shape: (batch_size,)
        best_count_number = torch.argmax(log_probs, -1)

        output_dict = {
            'log_probs': log_probs,
            'logits': logits,
            'best_count_number': best_count_number
        }
        return output_dict

    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 log_probs: torch.LongTensor,
                                 number_indices: torch.LongTensor,
                                 **kwargs: Any):
        answer_as_counts = gold_answer_representations['answer_as_counts']

        # Count answers are padded with label -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        # Shape: (batch_size, # of count answers)
        gold_count_mask = (answer_as_counts != -1).long()
        # Shape: (batch_size, # of count answers)
        clamped_gold_counts = replace_masked_values(answer_as_counts, gold_count_mask, 0)
        log_likelihood_for_counts = torch.gather(log_probs, 1, clamped_gold_counts)
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_counts = \
            replace_masked_values(log_likelihood_for_counts, gold_count_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood = logsumexp(log_likelihood_for_counts)
        return log_marginal_likelihood

    def decode_answer(self,
                      best_count_number: torch.LongTensor,
                      **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        predicted_count = best_count_number.detach().cpu().numpy().tolist()
        predicted_answer = str(predicted_count)

        answer_dict = {
            'value': predicted_answer,
            'count': predicted_count
        }
        return answer_dict
