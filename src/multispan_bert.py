from typing import Dict, Optional, List, Any, Tuple
from overrides import overrides

import torch
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from pytorch_pretrained_bert import BertModel, BertTokenizer
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1

import numpy as np

@Model.register('multi_span_bert')
class MultiSpanBert(Model):
    '''
    A model that predicts answers to questions where the answer is a list of spans from the passage.
    '''
    def __init__(self, 
                 vocab: Vocabulary, 
                 bert_pretrained_model: str, 
                 dropout_prob: float = 0.1, 
                 finetune_bert: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.answering_abilities = ["multiple_span"]
                
        self.BERT = BertModel.from_pretrained(bert_pretrained_model)
        
        if not finetune_bert:
            self.BERT.eval()

        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        bert_dim = self.BERT.pooler.dense.out_features
        
        self.dropout = dropout_prob

        self._passage_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(bert_dim, 1)
            
        self.multi_span_predictor = self.ff(bert_dim, bert_dim, 3)

        include_start_end_transitions = True
        constraints = allowed_transitions('BIO', {0: 'O', 1: 'B', 2: 'I'})

        self.crf = ConditionalRandomField(3, constraints, include_start_end_transitions)

        self._drop_metrics = DropEmAndF1()
        initializer(self)

    def forward(self,
                question_and_passage: Dict[str, torch.LongTensor],
                span_labels: torch.LongTensor = None,
                span_wordpiece_mask: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        
        # Shape: (batch_size, seqlen)
        question_passage_tokens = question_and_passage["tokens"]
        
        # Shape: (batch_size, seqlen)
        pad_mask = question_and_passage["mask"] 

        # Shape: (batch_size, seqlen)
        seqlen_ids = question_and_passage["tokens-type-ids"]

        mask = pad_mask

        if span_wordpiece_mask is not None:
            mask = span_wordpiece_mask & mask

        max_seqlen = question_passage_tokens.shape[-1]
        batch_size = question_passage_tokens.shape[0]

        # Shape: (batch_size, seqlen, bert_dim)
        bert_out, _ = self.BERT(question_passage_tokens, seqlen_ids, pad_mask, output_all_encoded_layers=False)

        logits = self.multi_span_predictor(bert_out)

        predicted_tags_with_score = self.crf.viterbi_tags(logits, pad_mask) #TODO does it ignore the mask?

        predicted_tags = [x for x, y in predicted_tags_with_score]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if span_labels is not None:
            log_likelihood = self.crf(logits, span_labels, mask)
            output["loss"] = -log_likelihood

        with torch.no_grad():
            output['answer'] = []
            for i in np.arange(batch_size):
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:                
                    answer_spans = self.decode_spans_from_tags(predicted_tags[i],  metadata[i]['question_passage_tokens'])
                    output['answer'].append(answer_spans)
                    self._drop_metrics(answer_spans, answer_annotations)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}

    def ff(self, input_dim, hidden_dim, output_dim):
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(self.dropout),
                                    torch.nn.Linear(hidden_dim, output_dim))

    def decode_spans_from_tags(self, tags, question_passage_tokens):
        spans_tokens = []
        prev = 0 # 0 = O
            
        current_tokens = []

        for i in np.arange(len(tags)):      
            token = question_passage_tokens[i].text

            # If it is the same word so just add it to current tokens
            if token[:2] == '##':
                if prev != 0:
                    current_tokens.append(token)
                continue

            if tags[i] == 1: # 1 = B
                if prev != 0:
                    spans_tokens.append(current_tokens)
                    current_tokens = []

                current_tokens.append(token)
                prev = 1
                continue

            if tags[i] == 2: # 2 = I
                current_tokens.append(token)
                prev = 2
                continue

            if tags[i] == 0 and prev != 0:
                spans_tokens.append(current_tokens)
                current_tokens = []
                prev = 0

        if current_tokens:
            spans_tokens.append(current_tokens)

        spans = [tokenlist_to_passage(tokens) for tokens in spans_tokens]              

        return spans


def tokenlist_to_passage(token_text):
    str_list = list(map(lambda x : x[2:] if len(x)>2 and x[:2]=="##" else " " + x, token_text))
    string = "".join(str_list)
    if string[0] == " ":
        string = string[1:]
    return string

