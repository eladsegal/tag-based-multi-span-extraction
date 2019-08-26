from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn import util
import torch
import numpy as np

class MultiSpanHandler:
    def __init__(self, 
                 bert_dim: int,
                 multi_span_predictor: torch.nn.Sequential,
                 crf: ConditionalRandomField,
                 dropout_prob: float = 0.1) -> None:
        self.dropout = dropout_prob

        self.multi_span_predictor = multi_span_predictor        

        self.crf = crf

    def forward(self, bert_out, span_labels, pad_mask, span_wordpiece_mask):            
        loss_mask = pad_mask

        if span_wordpiece_mask is not None:
            loss_mask = span_wordpiece_mask & loss_mask

        non_bio_mask = None

        if span_labels is not None:
            non_bio_mask = torch.ones(loss_mask.shape[0], dtype=torch.long, device = bert_out.device)

            for i in np.arange(loss_mask.shape[0]):
                if span_labels[i].sum() <= 0:
                    non_bio_mask[i] = 0 

        logits = self.multi_span_predictor(bert_out)

        predicted_tags_with_score = self.crf.viterbi_tags(logits, pad_mask) 

        predicted_tags = [x for x, y in predicted_tags_with_score]

        result = {"logits": logits, "predicted_tags": predicted_tags}

        if span_labels is not None:
            log_denominator = self.crf._input_likelihood(logits, loss_mask)
            log_numerator = self.crf._joint_likelihood(logits, span_labels, loss_mask)

            log_likelihood = log_numerator - log_denominator
            
            log_likelihood = util.replace_masked_values(log_likelihood, non_bio_mask, -1e7)

            result["log_likelihood"] = log_likelihood
            result["loss"] = -torch.sum(log_likelihood)

        return result

    def decode_spans_from_tags(self, tags, question_passage_tokens, passage_text, question_text):
        spans_tokens = []
        prev = 0 # 0 = O
            
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

            if tags[i] == 1: # 1 = B
                if prev != 0:
                    spans_tokens.append((context, current_tokens))
                    current_tokens = []

                current_tokens.append(token)
                prev = 1
                continue

            if tags[i] == 2: # 2 = I
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

        spans_indices.append((context, text_start, text_end))

        if context == 'p':
            spans_text.append(passage_text[text_start:text_end])
        else:
            spans_text.append(question_text[text_start:text_end])

    return spans_text, spans_indices

def default_multispan_predictor(bert_dim, dropout):
    return ff(bert_dim, bert_dim, 3, dropout)

def default_crf():
    include_start_end_transitions = True
    constraints = allowed_transitions('BIO', {0: 'O', 1: 'B', 2: 'I'})
    return ConditionalRandomField(3, constraints, include_start_end_transitions)

def ff(input_dim, hidden_dim, output_dim, dropout):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(dropout),
                                torch.nn.Linear(hidden_dim, output_dim))