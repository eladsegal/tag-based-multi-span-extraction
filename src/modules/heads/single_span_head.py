from typing import Dict, Any, Union

import torch

from allennlp.nn.util import replace_masked_values, logsumexp, masked_log_softmax
from allennlp.modules import FeedForward

from src.modules.heads.head import Head
from src.modules.utils.decoding_utils import decode_token_spans

class SingleSpanHead(Head):
    def __init__(self,
                 start_output_layer: FeedForward,
                 end_output_layer: FeedForward,
                 training_style: str) -> None:
        super().__init__()
        self._start_output_layer = start_output_layer
        self._end_output_layer = end_output_layer
        self._training_style = training_style

    def forward(self,                
                **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        input, mask = self.get_input_and_mask(kwargs)

        # Shape: (batch_size, passage_length)
        start_logits = self._start_output_layer(input).squeeze(-1)

        # Shape: (batch_size, passage_length)
        end_logits = self._end_output_layer(input).squeeze(-1)

        start_log_probs = masked_log_softmax(start_logits, mask)
        end_log_probs = masked_log_softmax(end_logits, mask)

        # Info about the best span prediction
        start_logits = replace_masked_values(start_logits, mask, -1e7)
        end_logits = replace_masked_values(end_logits, mask, -1e7)

        # Shape: (batch_size, 2)
        best_span = get_best_span(start_logits, end_logits)

        output_dict = {
            'start_log_probs': start_log_probs,
            'end_log_probs': end_log_probs,
            'best_span': best_span
        }
        return output_dict

    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 start_log_probs: torch.LongTensor,
                                 end_log_probs: torch.LongTensor,
                                 **kwargs: Any):
        answer_as_spans = self.get_gold_answer_representations(gold_answer_representations)

        # Shape: (batch_size, # of answer spans)
        gold_span_starts = answer_as_spans[:, :, 0]
        gold_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_span_mask = (gold_span_starts != -1).long()
        clamped_gold_span_starts = \
            replace_masked_values(gold_span_starts, gold_span_mask, 0)
        clamped_gold_span_ends = \
            replace_masked_values(gold_span_ends, gold_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = \
            torch.gather(start_log_probs, 1, clamped_gold_span_starts)
        log_likelihood_for_span_ends = \
            torch.gather(end_log_probs, 1, clamped_gold_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = \
            log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = \
            replace_masked_values(log_likelihood_for_spans, gold_span_mask, -1e7)

        # Shape: (batch_size, )
        if self._training_style == 'soft_em':
            log_marginal_likelihood_for_span = logsumexp(log_likelihood_for_spans)
        elif self._training_style == 'hard_em':
            most_likely_span_index = log_likelihood_for_spans.argmax(dim=-1)
            log_marginal_likelihood_for_span = log_likelihood_for_spans.gather(dim=1, index=most_likely_span_index.unsqueeze(-1)).squeeze(dim=-1)
        else:
            raise Exception("Illegal training_style")

        return log_marginal_likelihood_for_span

    def decode_answer(self,
                      qp_tokens: torch.LongTensor,
                      best_span: torch.Tensor,
                      p_text: str,
                      q_text: str,
                      **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        
        (predicted_start, predicted_end)  = tuple(best_span.detach().cpu().numpy())
        answer_tokens = qp_tokens[predicted_start:predicted_end + 1]
        spans_text, spans_indices = decode_token_spans([(self.get_context(), answer_tokens)], p_text, q_text)
        predicted_answer = spans_text[0]

        answer_dict = {
            'value': predicted_answer,
            'spans': spans_indices
        }
      
        return answer_dict

    def get_input_and_mask(self, kwargs: Dict[str, Any]) -> torch.LongTensor:
        raise NotImplementedError

    def get_gold_answer_representations(self, gold_answer_representations: Dict[str, torch.LongTensor]) -> torch.LongTensor:
        raise NotImplementedError

    def get_context(self) -> str:
        raise NotImplementedError

def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)
