import numpy as np
import torch
from torch.nn import Module

from typing import Tuple, Dict

from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.nn.util import replace_masked_values, logsumexp, masked_log_softmax
from src.beam_search import BeamSearch

class MultiSpanHead(Module):
    def __init__(self,
                 bert_dim: int,
                 predictor: Module = None,
                 dropout: float = 0.1) -> None:
        super(MultiSpanHead, self).__init__()
        self.bert_dim = bert_dim
        self.dropout = dropout
        self.predictor = predictor or MultiSpanHead.default_predictor(self.bert_dim, self.dropout)

    def module(self):
        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError

    def prediction(self):
        raise NotImplementedError

    @staticmethod
    def default_predictor(bert_dim, dropout):
        return ff(bert_dim, bert_dim, 3, dropout)

    @staticmethod
    def decode_spans_from_tags(tags, question_passage_tokens, passage_text, question_text):
        spans_tokens = []
        prev = 0  # 0 = O

        current_tokens = []

        context = 'q'

        for i in np.arange(len(question_passage_tokens)):
            token = question_passage_tokens[i]

            if token.text == '[SEP]':
                context = 'p'

            # If it is the same word so just add it to current tokens
            if token.text[:2] == '##':
                if prev != 0:
                    current_tokens.append(token)
                continue

            if tags[i] == 1:  # 1 = B
                if prev != 0:
                    spans_tokens.append((context, current_tokens))
                    current_tokens = []

                current_tokens.append(token)
                prev = 1
                continue

            if tags[i] == 2:  # 2 = I
                if prev != 0:
                    current_tokens.append(token)
                    prev = 2
                else:
                    prev = 0 # Illegal I, treat it as 0
                '''# Written under the assumption that if it was meant for the BIO rules to be enforced
                # then it was already done for tags
                current_tokens.append(token)
                prev = 2'''
                continue

            if tags[i] == 0 and prev != 0:
                spans_tokens.append((context, current_tokens))
                current_tokens = []
                prev = 0

        if current_tokens:
            spans_tokens.append((context, current_tokens))

        valid_tokens, invalid_tokens = validate_tokens_spans(spans_tokens)
        spans_text, spans_indices = decode_token_spans(valid_tokens, passage_text, question_text)

        return spans_text, spans_indices, invalid_tokens, tags, [token.text for token in question_passage_tokens]


class SimpleBIO(MultiSpanHead):
    def __init__(self, bert_dim: int, predictor: Module = None, dropout_prob: float = 0.1) -> None:
        super(SimpleBIO, self).__init__(bert_dim, predictor, dropout_prob)

    def module(self, bert_out):
        logits = self.predictor(bert_out)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs, logits

    def log_likelihood(self, gold_labels, log_probs, seq_mask, is_bio_mask, **kwargs):

        # we only want the log probabilities of the gold labels
        # what we get is:
        # log_likelihoods_for_multispan[i,j] == log_probs[i,j, gold_labels[i,j]]
        log_likelihoods_for_multispan = \
            torch.gather(log_probs, dim=-1, index=gold_labels.unsqueeze(-1)).squeeze(-1)

        # Our marginal likelihood is the sum of all the gold label likelihoods, ignoring the
        # padding tokens.
        log_likelihoods_for_multispan = \
            replace_masked_values(log_likelihoods_for_multispan, seq_mask, 0.0)

        log_marginal_likelihood_for_multispan = log_likelihoods_for_multispan.sum(dim=-1)

        # For questions without spans, we set their log probabilities to be very small negative value
        log_marginal_likelihood_for_multispan = \
            replace_masked_values(log_marginal_likelihood_for_multispan, is_bio_mask, -1e7)

        return log_marginal_likelihood_for_multispan

    def prediction(self, log_probs, logits, qp_tokens, p_text, q_text, mask):

        predicted_tags = torch.argmax(logits, dim=-1)
        predicted_tags = replace_masked_values(predicted_tags, mask, 0)

        return MultiSpanHead.decode_spans_from_tags(predicted_tags, qp_tokens, p_text, q_text)


class CRFLossBIO(MultiSpanHead):

    def __init__(self, bert_dim: int, predictor: Module = None, dropout_prob: float = 0.1) -> None:
        super(CRFLossBIO, self).__init__(bert_dim, predictor, dropout_prob)

        # create crf for tag decoding
        self.crf = default_crf()

    def module(self, bert_out):
        logits = self.predictor(bert_out)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs, logits

    def log_likelihood(self, gold_labels, log_probs, seq_mask, is_bio_mask, **kwargs):

        logits = kwargs['logits']

        if gold_labels is not None:
            log_denominator = self.crf._input_likelihood(logits, seq_mask)
            log_numerator = self.crf._joint_likelihood(logits, gold_labels, seq_mask)

            log_likelihood = log_numerator - log_denominator

            log_likelihood = replace_masked_values(log_likelihood, is_bio_mask, -1e7)

            return log_likelihood

    def prediction(self, log_probs, logits, qp_tokens, p_text, q_text, mask):

        predicted_tags_with_score = self.crf.viterbi_tags(logits.unsqueeze(0), mask.unsqueeze(0))
        predicted_tags = [x for x, y in predicted_tags_with_score]

        return MultiSpanHead.decode_spans_from_tags(predicted_tags[0], qp_tokens, p_text, q_text)

class FlexibleLoss(MultiSpanHead):

    def __init__(self, bert_dim: int, predictor: Module = None, dropout_prob: float = 0.1, use_crf=False, generation_top_k=1000, prediction_beam_size=50) -> None:
        super(FlexibleLoss, self).__init__(bert_dim, predictor, dropout_prob)
        
        self.crf = default_crf() # SHOULD BE REMOVED, USED ONLY TO LOAD OLDER MODELS

        self._use_crf = use_crf
        if use_crf:
            self._crf = default_crf()
        else:
            self._start_index, self._end_index = 3, 4
            self._per_node_beam_size = 3
            self._generation_top_k = generation_top_k
            self._prediction_beam_size = prediction_beam_size

    def module(self, bert_out, seq_mask=None):
        logits = self.predictor(bert_out)

        if self._use_crf:
            # The mask should not be applied here when using CRF, but should be passed ot the CRF
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            if seq_mask is not None:
                log_probs =  replace_masked_values(torch.nn.functional.log_softmax(logits, dim=-1), seq_mask.unsqueeze(-1), 0.0)
                logits =  replace_masked_values(logits, seq_mask.unsqueeze(-1), -1e7)
            else:
                log_probs = torch.nn.functional.log_softmax(logits)

        return log_probs, logits

    def log_likelihood(self, answer_as_text_to_disjoint_bios, answer_as_list_of_bios, span_bio_labels, log_probs, 
                        logits, seq_mask, is_bio_mask, **kwargs):
        # answer_as_text_to_disjoint_bios - Shape: (batch_size, # of text answers, # of spans a for text answer, seq_length)
        # answer_as_list_of_bios - Shape: (batch_size, # of correct sequences, seq_length)
        # log_probs - Shape: (batch_size, seq_length, 3)
        # seq_mask - Shape: (batch_size, seq_length)
        
        # Generate most likely correct predictions
        if self._use_crf:
            raise NotImplementedError
        else:
            with torch.no_grad():
                if answer_as_text_to_disjoint_bios.sum() > 0:
                    #most_likely_predictions = self._generate_most_likely_predictions(log_probs)
                    #most_likely_correct_predictions = self._filter_correct_predictions(most_likely_predictions, answer_as_text_to_disjoint_bios)
                    #full_bio = span_bio_labels
                    #generated_list_of_bios = self._add_full_bio(most_likely_correct_predictions, full_bio)
                    is_pregenerated_answer_format_mask = (answer_as_list_of_bios.sum((1, 2)) > 0).long()
                    list_of_bios = torch.cat((answer_as_list_of_bios, (span_bio_labels * (1 - is_pregenerated_answer_format_mask).unsqueeze(-1)).unsqueeze(1)), dim=1)
                else:
                    list_of_bios = answer_as_list_of_bios

        ### Calculate log-likelihood from list_of_bios
        if self._use_crf:
            raise NotImplementedError
        else:
            log_marginal_likelihood_for_multispan = self._get_combined_likelihood(list_of_bios, log_probs)

        # For questions without spans, we set their log probabilities to be very small negative value
        log_marginal_likelihood_for_multispan = \
            replace_masked_values(log_marginal_likelihood_for_multispan, is_bio_mask, -1e7)

        return log_marginal_likelihood_for_multispan

    def prediction(self, log_probs, logits, qp_tokens, p_text, q_text, seq_mask):
        '''seq_length = log_probs.size()[0]
        batch_size = 1

        beam_search = BeamSearch(self._end_index, max_steps=seq_length, beam_size=self._prediction_beam_size, per_node_beam_size=self._per_node_beam_size)
        beam_log_probs = torch.nn.functional.pad(log_probs.unsqueeze(0), pad=(0, 2, 0, 0, 0, 0), value=-1e7) # add low log probabilites for start and end tags used in the beam search
        start_predictions = beam_log_probs.new_full((1,), fill_value=self._start_index).long()

        # Shape: (batch_size, beam_size, seq_length)
        top_k_predictions, seq_log_probs = beam_search.search(
                start_predictions, {'log_probs': beam_log_probs, 'step_num': beam_log_probs.new_zeros((batch_size,)).long()}, self.take_step)

        predicted_tags = top_k_predictions[0, 0, :]'''
        predicted_tags = torch.argmax(logits, dim=-1)
        predicted_tags = replace_masked_values(predicted_tags, seq_mask, 0)

        return MultiSpanHead.decode_spans_from_tags(predicted_tags, qp_tokens, p_text, q_text)

    def _generate_most_likely_predictions(self, log_probs):
        batch_size = log_probs.size()[0]
        seq_length = log_probs.size()[1]

        beam_search = BeamSearch(self._end_index, max_steps=seq_length, beam_size=self._generation_top_k, per_node_beam_size=self._per_node_beam_size)
        beam_log_probs = torch.nn.functional.pad(log_probs, pad=(0, 2, 0, 0, 0, 0), value=-1e7) # add low log probabilites for start and end tags used in the beam search
        start_predictions = beam_log_probs.new_full((batch_size,), fill_value=self._start_index).long()

        # Shape: (batch_size, beam_size, seq_length)
        top_k_predictions, seq_log_probs = beam_search.search(
                start_predictions, {'log_probs': beam_log_probs, 'step_num': beam_log_probs.new_zeros((batch_size,)).long()}, self.take_step)

        return top_k_predictions

    def _filter_correct_predictions(self, predictions, answer_as_text_to_disjoint_bios):
        texts_count = answer_as_text_to_disjoint_bios.size()[1]
        spans_count = answer_as_text_to_disjoint_bios.size()[2]
        predictions_count = predictions.size()[1]

        expanded_predictions = predictions.unsqueeze(2).unsqueeze(2).repeat(1, 1, texts_count, spans_count, 1)
        expanded_answer_as_text_to_disjoint_bios = answer_as_text_to_disjoint_bios.unsqueeze(1).repeat(1, predictions_count, 1, 1, 1)

        correct_mask = ((((expanded_predictions == expanded_answer_as_text_to_disjoint_bios) & (expanded_answer_as_text_to_disjoint_bios != 0)).sum(-2).sum(-1) > 0).prod(-1) != 0).long()

        return predictions * correct_mask.unsqueeze(-1)

    def _add_full_bio(self, most_likely_correct_predictions, full_bio):
        predictions_count = most_likely_correct_predictions.size()[1]

        already_added = ((full_bio.unsqueeze(1).repeat(1, predictions_count, 1) == most_likely_correct_predictions).sum((-1,-2)) == 0).long()

        return torch.cat((most_likely_correct_predictions, (full_bio * already_added.unsqueeze(-1)).unsqueeze(1)), dim=1)

    def _get_combined_likelihood(self, answer_as_list_of_bios, log_probs):
        # answer_as_list_of_bios - Shape: (batch_size, # of correct sequences, seq_length)
        # log_probs - Shape: (batch_size, seq_length, 3)

        # Shape: (batch_size, # of correct sequences, seq_length, 3)
        # duplicate log_probs for each gold bios sequence
        expanded_log_probs = log_probs.unsqueeze(1).expand(-1, answer_as_list_of_bios.size()[1], -1, -1)
        
        # get the log-likelihood per each sequence index
        # Shape: (batch_size, # of correct sequences, seq_length)
        log_likelihoods = \
            torch.gather(expanded_log_probs, dim=-1, index=answer_as_list_of_bios.unsqueeze(-1)).squeeze(-1)

        # Shape: (batch_size, # of correct sequences)
        correct_sequences_pad_mask = (answer_as_list_of_bios.sum(-1) > 0).long()

        # Sum the log-likelihoods for each index to get the log-likelihood of the sequence
        # Shape: (batch_size, # of correct sequences)
        sequences_log_likelihoods = log_likelihoods.sum(dim=-1)
        sequences_log_likelihoods = replace_masked_values(sequences_log_likelihoods, correct_sequences_pad_mask, -1e7)

        # Sum the log-likelihoods for each sequence to get the marginalized log-likelihood over the correct answers
        log_marginal_likelihood = logsumexp(sequences_log_likelihoods, dim=-1)

        return log_marginal_likelihood

    def take_step(self, last_predictions: torch.Tensor,
                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # get the relevant scores for the time step
        class_log_probabilities = state['log_probs'][:,state['step_num'][0],:]

        # mask illegal BIO transitions
        transitions_mask = torch.ones_like(class_log_probabilities).byte()
        transitions_mask[:,2] &= ((last_predictions != 1) & (last_predictions != 2))
        transitions_mask[:,1:] &= ((class_log_probabilities[:,:3] == 0.0).sum(-1) != 3).unsqueeze(-1).repeat(1,4)
        class_log_probabilities = replace_masked_values(class_log_probabilities, transitions_mask, -1e7)

        state['step_num'] = state['step_num'].clone() + 1
        return  class_log_probabilities, state


multispan_heads_mapping = {'simple_bio': SimpleBIO, 'crf_loss_bio': CRFLossBIO, 'flexible_loss': FlexibleLoss}


def ff(input_dim, hidden_dim, output_dim, dropout):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(dropout),
                               torch.nn.Linear(hidden_dim, output_dim))


def validate_tokens_spans(spans_tokens):
    valid_tokens = []
    invalid_tokens = []
    for context, tokens in spans_tokens:
        tokens_text = [token.text for token in tokens]

        if '[CLS]' in tokens_text or '[SEP]' in tokens_text:
            invalid_tokens.append(tokens)
        else:
            valid_tokens.append((context, tokens))

    return valid_tokens, invalid_tokens


def decode_token_spans(spans_tokens, passage_text, question_text):
    spans_text = []
    spans_indices = []

    for context, tokens in spans_tokens:
        text_start = tokens[0].idx
        text_end = tokens[-1].idx + len(tokens[-1].text)

        if tokens[-1].text.startswith("##"):
            text_end -= 2

        if tokens[-1].text == '[UNK]':
            text_end -= 4

        spans_indices.append((context, text_start, text_end))

        if context == 'p':
            spans_text.append(passage_text[text_start:text_end])
        else:
            spans_text.append(question_text[text_start:text_end])

    return spans_text, spans_indices


def default_crf() -> ConditionalRandomField:
    include_start_end_transitions = True
    constraints = allowed_transitions('BIO', {0: 'O', 1: 'B', 2: 'I'})
    return ConditionalRandomField(3, constraints, include_start_end_transitions)