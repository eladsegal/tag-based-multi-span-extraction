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
from allennlp.tools.drop_eval import answer_json_to_strings

import numpy as np

from src.multispan_handler import MultiSpanHandler, default_multispan_predictor, default_crf

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

        self._multispan_predictor = default_multispan_predictor(bert_dim, dropout_prob)
        self._multispan_crf = default_crf()
        self.multi_span_handler = MultiSpanHandler(bert_dim, self._multispan_predictor, self._multispan_crf, dropout_prob)

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

        max_seqlen = question_passage_tokens.shape[-1]
        batch_size = question_passage_tokens.shape[0]

        # Shape: (batch_size, seqlen, bert_dim)
        bert_out, _ = self.BERT(question_passage_tokens, seqlen_ids, pad_mask, output_all_encoded_layers=False)

        multi_span_result = self.multi_span_handler.forward(bert_out, span_labels, pad_mask, span_wordpiece_mask)

        predicted_tags = multi_span_result['predicted_tags']
        
        #output = {"logits": logits, "mask": mask, "tags": predicted_tags}
        output = dict()

        if "loss" in multi_span_result:
            output["loss"] = multi_span_result["loss"]

        with torch.no_grad():
            output["passage_id"] = []
            output["query_id"] = []
            output['answer'] = []
            output['ground_truth'] = []
            for i in np.arange(batch_size):
                output["passage_id"].append(metadata[i]["passage_id"])
                output["query_id"].append(metadata[i]["question_id"])
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:                
                    answer_value, answer_spans = self.multi_span_handler.decode_spans_from_tags(predicted_tags[i],  metadata[i]['question_passage_tokens'])
                    output['answer'].append(answer_value)                    
                    self._drop_metrics(answer_value, answer_annotations)

                    output['ground_truth'].append([answer_json_to_strings(annotation)[0] for annotation in answer_annotations])

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
