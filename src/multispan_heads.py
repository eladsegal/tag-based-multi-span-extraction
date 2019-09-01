from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.nn.util import replace_masked_values
import numpy as np
import torch
from torch.nn import Module


class MultiSpanHead(Module):
    def __init__(self,
                 bert_dim: int,
                 predictor: Module = None,
                 dropout_prob: float = 0.1) -> None:
        super(MultiSpanHead, self).__init__()
        self.bert_dim = bert_dim
        self.dropout = dropout_prob
        self.predictor = predictor or MultiSpanHead.default_predictor(self.bert_dim, self.dropout)

    def module(self, bert_out):
        raise NotImplementedError

    def log_likelihood(self, gold_labels, log_probs, likelihood_mask, is_bio_mask):
        raise NotImplementedError

    def prediction(self, logits, qp_tokens, p_text, q_text, mask):
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

        for i in np.arange(len(tags)):
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
                current_tokens.append(token)
                prev = 2
                continue

            if tags[i] == 0 and prev != 0:
                spans_tokens.append((context, current_tokens))
                current_tokens = []
                prev = 0

        if current_tokens:
            spans_tokens.append((context, current_tokens))

        valid_tokens, invalid_tokens = validate_tokens_spans(spans_tokens)
        spans_text, spans_indices = decode_token_spans(valid_tokens, passage_text, question_text)

        return spans_text, spans_indices, invalid_tokens


class SimpleBIO(MultiSpanHead):
    def __init__(self, bert_dim: int, predictor: Module = None, dropout_prob: float = 0.1) -> None:
        super(SimpleBIO, self).__init__(bert_dim, predictor, dropout_prob)

        # create crf for tag decoding
        self.crf = default_crf()

    def module(self, bert_out):
        logits = self.predictor(bert_out)  # .squeeze(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Unlike other `_module` methods (e.g `_passage_span_module`), we do not return our best
        # prediction here, just the log probabilities and the logits. This is a speed issue:
        # decoding a valid BIO sequence from the logits requires using viterbi, which makes the
        # training much slower. Instead, we choose to decode the BIO labels only during evaluation.
        # We are aware that this impacts any metric computed on the training set.
        return log_probs, logits

    # sum version
    def log_likelihood(self, gold_labels, log_probs, likelihood_mask, is_bio_mask, **kwargs):

        # take the log probability only for the gold labels
        log_likelihoods_for_multispan = \
            torch.gather(log_probs, dim=-1, index=gold_labels.unsqueeze(-1)).squeeze(-1)

        # Our marginal likelihood is the sum of all the gold label likelihoods, ignoring the [CLS]
        # and [SEP] tokens, as well as the padding tokens.
        log_likelihoods_for_multispan = \
            replace_masked_values(log_likelihoods_for_multispan, gold_labels != 0, 0.0)

        log_marginal_likelihood_for_multispan = log_likelihoods_for_multispan.sum(dim=-1)

        # For questions without spans, we set their log probabilities to be very small negative value
        has_spans_mask = gold_labels.sum(dim=-1) != 0
        log_marginal_likelihood_for_multispan = \
            replace_masked_values(log_marginal_likelihood_for_multispan, has_spans_mask, -1e7)

        return log_marginal_likelihood_for_multispan

    # # mean version
    # def log_likelihood(self, gold_labels, log_probs, mask):
    #
    #     # take the lob probability only for the gold labels
    #     log_likelihoods_for_multispan = \
    #         torch.gather(log_probs, dim=-1, index=gold_labels.unsqueeze(-1)).squeeze()
    #
    #     # Our marginal likelihood is the mean of all the gold label likelihoods, ignoring the [CLS]
    #     # and [SEP] tokens, as well as the padding tokens.
    #     log_marginal_likelihood_for_multispan = masked_mean(log_likelihoods_for_multispan, mask, dim=-1)
    #
    #     # In a heuristic way, we take the square of the marginal likelihood to be in the magnitude of
    #     # the multiplication of two probabilities, as in the single span heads for a single span
    #     # question. The arbitrariness of this choice gets clearer when we remember that in multi span
    #     # answers for single span heads, the marginal likelihood is the sum of all the marginal span
    #     # probabilities, i.e a sum of (multiplication of two probabilities).
    #     return log_marginal_likelihood_for_multispan * 2

    def prediction(self, logits, qp_tokens, p_text, q_text, mask):

        predicted_tags_with_score = self.crf.viterbi_tags(logits.unsqueeze(0), mask.unsqueeze(0))
        predicted_tags = [x for x, y in predicted_tags_with_score]

        return MultiSpanHead.decode_spans_from_tags(predicted_tags[0],  qp_tokens, p_text, q_text)


class CRFLossBIO(MultiSpanHead):

    def __init__(self, bert_dim: int, predictor: Module = None, dropout_prob: float = 0.1) -> None:
        super(CRFLossBIO, self).__init__(bert_dim, predictor, dropout_prob)

        # create crf for tag decoding
        self.crf = default_crf()

    def module(self, bert_out):
        logits = self.predictor(bert_out)
        # Unlike non-multispan `_module` methods (e.g `_passage_span_module`), we do not return our
        # best prediction here, just the log probabilities and the logits. This is a speed issue:
        # decoding a valid BIO sequence from the logits requires using viterbi, which makes the
        # training much slower. Instead, we choose to decode the BIO labels only during evaluation.
        # We are aware that this impacts any metric computed on the training set.

        # In addition, all other `_module` methods, (including the other multispan ones), are
        # expected to return the log probabilities, not the log likelihoods. In all other heads,
        # first you calculate the logits, then from the logits you calculate the log probabilities,
        # and you return them. In turn, the `_log_likelihood` method use the log probabilities to
        # compute the log likelihood, and return it.
        # However, as this head uses the crf loss, and as the `allennlp` crf implementation computes
        # the log likelihood directly (skipping the computation of the log probabilities).
        # This is why here we return the log probabilities as `None`: they are not used in this class
        log_probs = None
        return log_probs, logits

    def log_likelihood(self, gold_labels, log_probs, likelihood_mask, is_bio_mask, **kwargs):

        logits = kwargs['logits']

        if gold_labels is not None:
            log_denominator = self.crf._input_likelihood(logits, likelihood_mask)
            log_numerator = self.crf._joint_likelihood(logits, gold_labels, likelihood_mask)

            log_likelihood = log_numerator - log_denominator

            log_likelihood = replace_masked_values(log_likelihood, is_bio_mask, -1e7)

            return log_likelihood

    def prediction(self, logits, qp_tokens, p_text, q_text, mask):

        predicted_tags_with_score = self.crf.viterbi_tags(logits.unsqueeze(0), mask.unsqueeze(0))
        predicted_tags = [x for x, y in predicted_tags_with_score]

        return MultiSpanHead.decode_spans_from_tags(predicted_tags[0], qp_tokens, p_text, q_text)


multispan_heads_mapping = {'simple_bio': SimpleBIO, 'crf_loss_bio': CRFLossBIO}


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