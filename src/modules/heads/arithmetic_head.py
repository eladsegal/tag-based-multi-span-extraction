from typing import Dict, Any, List, Union

import torch

from allennlp.nn.util import replace_masked_values, logsumexp
from allennlp.modules import FeedForward

from src.modules.heads.head import Head

@Head.register('arithmetic_head')
class ArithmeticHead(Head):
    def __init__(self,
                 output_layer: FeedForward,
                 special_numbers: List[Union[int, float]],
                 special_embedding_dim: int,
                 training_style: str,
                 arithmetic_round_ndigits: int = 5) -> None:
        super().__init__()
        self._output_layer = output_layer
        self._special_numbers = special_numbers
        self._training_style = training_style
        self._arithmetic_round_ndigits = arithmetic_round_ndigits

        self._num_special_numbers = len(self._special_numbers)
        self._special_embeddings = torch.nn.Embedding(self._num_special_numbers, special_embedding_dim)

    def forward(self,
                token_representations: torch.LongTensor,
                passage_summary_vector: torch.LongTensor,
                number_indices: torch.LongTensor,
                **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:

        number_mask = self._get_mask(number_indices, with_special_numbers=False)

        clamped_number_indices = replace_masked_values(number_indices[:,:,0].long(), number_mask, 0)
        encoded_numbers = torch.gather(
                token_representations,
                1,
                clamped_number_indices.unsqueeze(-1).expand(-1, -1, token_representations.size(-1)))
            
        if self._num_special_numbers > 0:
            special_numbers = self._special_embeddings(torch.arange(self._num_special_numbers, device=number_indices.device))
            special_numbers = special_numbers.expand(number_indices.shape[0],-1,-1)
            encoded_numbers = torch.cat([special_numbers, encoded_numbers], 1)
            
        # Shape: (batch_size, # of numbers, 2*bert_dim)
        encoded_numbers = torch.cat(
                [encoded_numbers, passage_summary_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

        # Shape: (batch_size, # of numbers in the passage, 3)
        logits = self._output_layer(encoded_numbers)
        log_probs = torch.nn.functional.log_softmax(logits, -1)

        number_mask = self._get_mask(number_indices, with_special_numbers=True)
        # Shape: (batch_size, # of numbers in passage).
        best_signs_for_numbers = torch.argmax(log_probs, -1)
        # For padding numbers, the best sign masked as 0 (not included).
        best_signs_for_numbers = replace_masked_values(best_signs_for_numbers, number_mask, 0)

        output_dict = {
            'log_probs': log_probs,
            'logits': logits,
            'best_signs_for_numbers': best_signs_for_numbers
        }
        return output_dict

    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 log_probs: torch.LongTensor,
                                 number_indices: torch.LongTensor,
                                 **kwargs: Any):
        answer_as_expressions_extra = gold_answer_representations['answer_as_expressions_extra']
        answer_as_expressions = gold_answer_representations['answer_as_expressions']

        number_mask = self._get_mask(number_indices, with_special_numbers=True)

        if self._num_special_numbers > 0:
            answer_as_expressions = torch.cat([answer_as_expressions_extra, answer_as_expressions], -1)
        # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
        # Shape: (batch_size, # of combinations)
        gold_add_sub_mask = (answer_as_expressions.sum(-1) > 0).float()
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        gold_add_sub_signs = answer_as_expressions.transpose(1, 2)
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        log_likelihood_for_number_signs = torch.gather(log_probs, 2, gold_add_sub_signs)
        # the log likelihood of the masked positions should be 0
        # so that it will not affect the joint probability
        log_likelihood_for_number_signs = \
            replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
        # Shape: (batch_size, # of combinations)
        log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
        # For those padded combinations, we set their log probabilities to be very small negative value
        log_likelihood_for_add_subs = \
            replace_masked_values(log_likelihood_for_add_subs, gold_add_sub_mask, -1e7)
        
        # Shape: (batch_size,)
        if self._training_style == 'soft_em':
            log_marginal_likelihood = logsumexp(log_likelihood_for_add_subs)
        elif self._training_style == 'hard_em':
            most_likely_add_sub_index = log_likelihood_for_add_subs.argmax(dim=-1)
            log_marginal_likelihood = log_likelihood_for_add_subs.gather(dim=1, index=most_likely_add_sub_index.unsqueeze(-1)).squeeze(dim=-1)
        else:
            raise Exception("Illegal training_style")

        return log_marginal_likelihood

    def decode_answer(self,
                      original_numbers: List[Union[int, float]],
                      number_indices: torch.LongTensor,
                      best_signs_for_numbers: torch.LongTensor,
                      **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        sign_remap = {0: 0, 1: 1, 2: -1}
        original_numbers = self._special_numbers + original_numbers
        predicted_signs = [sign_remap[it] for it in best_signs_for_numbers.detach().cpu().numpy()]
        result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
        predicted_answer = str(round(result, self._arithmetic_round_ndigits))
        numbers = []
        for i, (value, sign) in enumerate(zip(original_numbers, predicted_signs)):
            numbers.append({
                'value': value, 
                'sign': sign, 
                'is_special': i < len(self._special_numbers)
            })
        if number_indices[-1][0] == -1:
            # There is a dummy 0 number at position -1 added in some cases; we are
            # removing that here.
            numbers.pop()

        answer_dict = {
            'value': predicted_answer,
            'numbers': numbers
        }
        return answer_dict

    def _get_mask(self, number_indices: torch.LongTensor, with_special_numbers: bool) -> torch.LongTensor:
        number_mask = (number_indices[:,:,0].long() != -1).long()
        if with_special_numbers and self._num_special_numbers > 0:
            mask = torch.ones((number_indices.shape[0], self._num_special_numbers), device=number_indices.device).long()
            number_mask = torch.cat([mask, number_mask], -1)

        return number_mask