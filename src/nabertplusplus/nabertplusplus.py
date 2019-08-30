from typing import Any, Dict, List, Optional
import logging

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from src.custom_drop_em_and_f1 import CustomDropEmAndF1
from allennlp.tools.drop_eval import answer_json_to_strings
from pytorch_transformers import BertModel
import pickle

from src.nhelpers import beam_search, evaluate_postfix

from src.multispan_handler import MultiSpanHandler, default_multispan_predictor, default_crf, decode_token_spans

logger = logging.getLogger(__name__)

@Model.register("nabert++")
class NumericallyAugmentedBERTPlusPlus(Model):
    """
    This class augments NABERT+ with multi span answering ability.
    The code is based on NABERT+ implementation.
    """
    def __init__(self, 
                 vocab: Vocabulary, 
                 bert_pretrained_model: str, 
                 dropout_prob: float = 0.1, 
                 max_count: int = 10,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None,
                 number_rep: str = 'first',
                 arithmetic: str = 'base',
                 special_numbers: List[int] = None,
                 unique_on_multispan: bool = True) -> None:
        super().__init__(vocab, regularizer)

        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "arithmetic", "counting", "multiple_spans"]
        else:
            self.answering_abilities = answering_abilities
        self.number_rep = number_rep

        self.BERT = BertModel.from_pretrained(bert_pretrained_model)
        bert_dim = self.BERT.pooler.dense.out_features
        
        self.dropout = dropout_prob

        self._passage_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._number_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._arithmetic_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._multispan_weights_predictor = torch.nn.Linear(bert_dim, 1)
            
        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = \
                self.ff(2 * bert_dim, bert_dim, len(self.answering_abilities))

        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._passage_span_start_predictor = torch.nn.Linear(bert_dim, 1)
            self._passage_span_end_predictor = torch.nn.Linear(bert_dim, 1)

        if "question_span_extraction" in self.answering_abilities:
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._question_span_start_predictor = self.ff(2 * bert_dim, bert_dim, 1)
            self._question_span_end_predictor = self.ff(2 * bert_dim, bert_dim, 1)

        if "arithmetic" in self.answering_abilities:
            self.arithmetic = arithmetic
            self._arithmetic_index = self.answering_abilities.index("arithmetic")
            self.special_numbers = special_numbers
            self.num_special_numbers = len(self.special_numbers)
            self.special_embedding = torch.nn.Embedding(self.num_special_numbers, bert_dim)
            if self.arithmetic == "base":
                self._number_sign_predictor = \
                    self.ff(2 * bert_dim, bert_dim, 3)
            else:
                self.init_arithmetic(bert_dim, bert_dim, bert_dim, layers=2, dropout=dropout_prob)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = self.ff(bert_dim, bert_dim, max_count + 1)  # `+1` for 0

        if "multiple_spans" in self.answering_abilities:
            self._multispan_predictor = default_multispan_predictor(bert_dim, dropout_prob)
            self._multispan_crf = default_crf()
            self._multi_span_handler = MultiSpanHandler(bert_dim, self._multispan_predictor, self._multispan_crf, dropout_prob)
            self._unique_on_multispan = unique_on_multispan

        self._drop_metrics = CustomDropEmAndF1()
        initializer(self)

    def summary_vector(self, encoding, mask, in_type="passage"):
        """
        In NABERT (and in NAQANET), a 'summary_vector' is created for some entities, such as the
        passage or the question. This vector is created as a weighted sum of the elements of the
        entity, e.g. the passage summary vector is a weighted sum of the passage tokens.

        The specific weighting for every entity type is a learned.

        Parameters
        ----------
        encoding : BERT's output layer
        mask : a Tensor with 1s only at the positions relevant to ``in_type``
        in_type : the entity we want to summarize, e.g the passage

        Returns
        -------
        The summary vector according to ``in_type``.
        """
        if in_type == "passage":
            # Shape: (batch_size, seqlen)
            alpha = self._passage_weights_predictor(encoding).squeeze()
        elif in_type == "question":
            # Shape: (batch_size, seqlen)
            alpha = self._question_weights_predictor(encoding).squeeze()
        elif in_type == "arithmetic":
            # Shape: (batch_size, seqlen)
            alpha = self._arithmetic_weights_predictor(encoding).squeeze()
        elif in_type == "multiple_spans":
            #TODO: currenttly not using it...
            alpha = self._multispan_weights_predictor(encoding).squeeze()
        else:
            # Shape: (batch_size, #num of numbers, seqlen)
            alpha = torch.zeros(encoding.shape[:-1], device=encoding.device)
            if self.number_rep == 'attention':
                alpha = self._number_weights_predictor(encoding).squeeze()     
        # Shape: (batch_size, seqlen) 
        # (batch_size, #num of numbers, seqlen) for numbers
        alpha = masked_softmax(alpha, mask)
        # Shape: (batch_size, out)
        # (batch_size, #num of numbers, out) for numbers
        h = util.weighted_sum(encoding, alpha)
        return h
    
    def ff(self, input_dim, hidden_dim, output_dim):
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(self.dropout),
                                   torch.nn.Linear(hidden_dim, output_dim))
    
    def init_arithmetic(self, bert_dim, input_dim=300, rnn_hidden_dim=500, layers=1, dropout=0.3):
        lstm_args = {'input_size' : input_dim, 'hidden_size' : rnn_hidden_dim, 'num_layers' : layers, 'batch_first' : True,
                    'dropout':dropout}
        self.rnn = torch.nn.LSTM(**lstm_args)
        self.rnndropout = torch.nn.Dropout(dropout)  
        
        self.Wst = torch.nn.Linear(rnn_hidden_dim, bert_dim)
        self.Wo = torch.nn.Linear(bert_dim, bert_dim)
        self.Wc = torch.nn.Linear(bert_dim, bert_dim)
        
        self.Wcon = torch.nn.Linear(bert_dim, bert_dim)
        
        self.arith_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        
        self.max_explen = 10
        self.num_ops = 5
        self.arithmetic_K = 5
        self.op_embeddings = torch.nn.Embedding(self.num_ops + 1,bert_dim)
        self.ops = ["END", "+", "-", "*", "/", "100"]
        self.END = 0

    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                mask_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_expressions: torch.LongTensor = None,
                answer_as_expressions_extra: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                span_bio_labels: torch.LongTensor = None,
                bio_wordpiece_mask: torch.LongTensor = None,
                is_bio_mask: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # Shape: (batch_size, seqlen)
        question_passage_tokens = question_passage["tokens"]
        # Shape: (batch_size, seqlen)
        pad_mask = question_passage["mask"] 
        # Shape: (batch_size, seqlen)
        seqlen_ids = question_passage["tokens-type-ids"]
        
        max_seqlen = question_passage_tokens.shape[-1]
        batch_size = question_passage_tokens.shape[0]
                
        # Shape: (batch_size, 3)
        mask = mask_indices.squeeze(-1)
        # Shape: (batch_size, seqlen)
        cls_sep_mask = \
            torch.ones(pad_mask.shape, device=pad_mask.device).long().scatter(1, mask, torch.zeros(mask.shape, device=mask.device).long())
        # Shape: (batch_size, seqlen)
        passage_mask = seqlen_ids * pad_mask * cls_sep_mask
        # Shape: (batch_size, seqlen)
        question_mask = (1 - seqlen_ids) * pad_mask * cls_sep_mask
        
        # Shape: (batch_size, seqlen, bert_dim)
        bert_out, _ = self.BERT(question_passage_tokens, seqlen_ids, pad_mask)
        # Shape: (batch_size, qlen, bert_dim)
        question_end = max(mask[:,1])
        question_out = bert_out[:,:question_end]
        # Shape: (batch_size, qlen)
        question_mask = question_mask[:,:question_end]
        # Shape: (batch_size, out)
        question_vector = self.summary_vector(question_out, question_mask, "question")
        
        passage_out = bert_out
        del bert_out
        
        # Shape: (batch_size, bert_dim)
        passage_vector = self.summary_vector(passage_out, passage_mask)
        
        if "arithmetic" in self.answering_abilities and self.arithmetic == "advanced":
            arithmetic_summary = self.summary_vector(passage_out, pad_mask, "arithmetic")
#             arithmetic_summary = self.summary_vector(question_out, question_mask, "arithmetic")
            
            # Shape: (batch_size, # of numbers in the passage)
            if number_indices.dim() == 3:
                number_indices = number_indices[:,:,0].long()
            number_mask = (number_indices != -1).long()
            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
            encoded_numbers = torch.gather(
                    passage_out,
                    1,
                    clamped_number_indices.unsqueeze(-1).expand(-1, -1, passage_out.size(-1)))
            op_mask = torch.ones((batch_size, self.num_ops + 1), device=number_mask.device).long()
            options_mask = torch.cat([op_mask, number_mask], -1)
            ops = self.op_embeddings(torch.arange(self.num_ops + 1, device=number_mask.device).expand(batch_size,-1))
            options = torch.cat([self.Wo(ops), self.Wc(encoded_numbers)], 1)

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._answer_ability_predictor(torch.cat([passage_vector, question_vector], -1))
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

        if "counting" in self.answering_abilities:
            count_number_log_probs, best_count_number = self._count_module(passage_vector)

        if "passage_span_extraction" in self.answering_abilities:
            passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span = \
                self._passage_span_module(passage_out, passage_mask)

        if "question_span_extraction" in self.answering_abilities:
            question_span_start_log_probs, question_span_end_log_probs, best_question_span = \
                self._question_span_module(passage_vector, question_out, question_mask)

        if "multiple_spans" in self.answering_abilities:
            multi_span_result = self._multi_span_handler.forward(passage_out, span_bio_labels, pad_mask, bio_wordpiece_mask, is_bio_mask)
            
        if "arithmetic" in self.answering_abilities:
            if self.arithmetic == "base":
                number_mask = (number_indices[:,:,0].long() != -1).long()
                number_sign_log_probs, best_signs_for_numbers, number_mask = \
                    self._base_arithmetic_module(passage_vector, passage_out, number_indices, number_mask)
            else:
                arithmetic_logits, best_expression = \
                    self._adv_arithmetic_module(arithmetic_summary, self.max_explen, options, options_mask, \
                                                   passage_out, pad_mask)
                shapes = arithmetic_logits.shape
                if (1-(arithmetic_logits != arithmetic_logits)).sum() != (shapes[0]*shapes[1]*shapes[2]):
                    print("bad logits")
                    arithmetic_logits = torch.rand(shapes, device = arithmetic_logits.device, requires_grad = True)
            
        output_dict = {}
        del passage_out, question_out
        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or answer_as_question_spans is not None \
                or answer_as_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    log_marginal_likelihood_for_passage_span = \
                        self._passage_span_log_likelihood(answer_as_passage_spans,
                                                          passage_span_start_log_probs,
                                                          passage_span_end_log_probs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":
                    log_marginal_likelihood_for_question_span = \
                        self._question_span_log_likelihood(answer_as_question_spans,
                                                           question_span_start_log_probs,
                                                           question_span_end_log_probs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)

                elif answering_ability == "arithmetic":
                    if self.arithmetic == "base":
                        log_marginal_likelihood_for_arithmetic = \
                            self._base_arithmetic_log_likelihood(answer_as_expressions,
                                                                 number_sign_log_probs,
                                                                 number_mask, 
                                                                 answer_as_expressions_extra)
                    else:
                        max_explen = answer_as_expressions.shape[-1]
                        possible_exps = answer_as_expressions.shape[1]
                        limit = min(possible_exps, 1000)
                        log_marginal_likelihood_for_arithmetic = \
                            self._adv_arithmetic_log_likelihood(arithmetic_logits[:,:max_explen,:], 
                                                                answer_as_expressions[:,:limit,:].long())
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_arithmetic)

                elif answering_ability == "counting":
                    log_marginal_likelihood_for_count = \
                        self._count_log_likelihood(answer_as_counts, 
                                                   count_number_log_probs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                elif answering_ability == "multiple_spans":
                    log_marginal_likelihood_list.append(multi_span_result["log_likelihood"])
                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
        
            output_dict["loss"] = - marginal_log_likelihood.mean()
        with torch.no_grad():
            # Compute the metrics and add the tokenized input to the output.
            if metadata is not None:
                if not self.training:
                    output_dict["passage_id"] = []
                    output_dict["query_id"] = []
                    output_dict["answer"] = []
                    output_dict["predicted_ability"] = []
                    output_dict["maximizing_ground_truth"] = []
                    output_dict["em"] = []
                    output_dict["f1"] = []
                    output_dict["invalid_spans"] = []    
                    output_dict["is_clipped"] = []                             

                for i in range(batch_size):
                    if len(self.answering_abilities) > 1:
                        predicted_ability_str = self.answering_abilities[best_answer_ability[i]]
                    else:
                        predicted_ability_str = self.answering_abilities[0]
                    
                    answer_json: Dict[str, Any] = {}

                    invalid_spans = []
                    
                    # We did not consider multi-mention answers here
                    if predicted_ability_str == "passage_span_extraction":
                        answer_json["answer_type"] = "passage_span"
                        answer_json["value"], answer_json["spans"] = \
                            self._span_prediction(metadata[i]['question_passage_tokens'], best_passage_span[i], metadata[i]['original_passage'], metadata[i]['original_question'], 'p')
                    elif predicted_ability_str == "question_span_extraction":
                        answer_json["answer_type"] = "question_span"
                        answer_json["value"], answer_json["spans"] = \
                            self._span_prediction(metadata[i]['question_passage_tokens'], best_question_span[i], metadata[i]['original_passage'], metadata[i]['original_question'], 'q')
                    elif predicted_ability_str == "arithmetic":  # plus_minus combination answer
                        answer_json["answer_type"] = "arithmetic"
                        original_numbers = metadata[i]['original_numbers']
                        if self.arithmetic == "base":
                            answer_json["value"], answer_json["numbers"] = \
                                self._base_arithmetic_prediction(original_numbers, number_indices[i], best_signs_for_numbers[i])
                        else:
                            answer_json["value"], answer_json["expression"] = \
                                self._adv_arithmetic_prediction(original_numbers, best_expression[i])
                    elif predicted_ability_str == "counting":
                        answer_json["answer_type"] = "count"
                        answer_json["value"], answer_json["count"] = \
                            self._count_prediction(best_count_number[i])

                    elif predicted_ability_str == "multiple_spans":
                        answer_json["answer_type"] = "multiple_spans"
                        answer_json["value"], answer_json["spans"], invalid_spans = self._multi_span_handler.decode_spans_from_tags(multi_span_result["predicted_tags"][i], metadata[i]['question_passage_tokens'], metadata[i]['original_passage'], metadata[i]['original_question'])
                        if self._unique_on_multispan:
                            answer_json["value"] = list(set(answer_json["value"]))
                    else:
                        raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")
                    
                    maximizing_ground_truth = None
                    em, f1 = None, None
                    answer_annotations = metadata[i].get('answer_annotations', [])
                    if answer_annotations:
                        (em, f1), maximizing_ground_truth = self._drop_metrics.call(answer_json["value"], answer_annotations, predicted_ability_str)

                    if not self.training:
                        output_dict["passage_id"].append(metadata[i]["passage_id"])
                        output_dict["query_id"].append(metadata[i]["question_id"])
                        output_dict["answer"].append(answer_json)
                        output_dict["predicted_ability"].append(predicted_ability_str)
                        output_dict["maximizing_ground_truth"].append(maximizing_ground_truth)
                        output_dict["em"].append(em)
                        output_dict["f1"].append(f1)
                        output_dict["invalid_spans"].append(invalid_spans)
                        output_dict["is_clipped"].append(metadata[i]["is_clipped"])

        return output_dict
    
    
    def _passage_span_module(self, passage_out, passage_mask):
        # Shape: (batch_size, passage_length)
        passage_span_start_logits = self._passage_span_start_predictor(passage_out).squeeze(-1)

        # Shape: (batch_size, passage_length)
        passage_span_end_logits = self._passage_span_end_predictor(passage_out).squeeze(-1)

        # Shape: (batch_size, passage_length)
        passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
        passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)

        # Info about the best passage span prediction
        passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask, -1e7)
        passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask, -1e7)

        # Shape: (batch_size, 2)
        best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)
        return passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span
    
    
    def _passage_span_log_likelihood(self,
                                     answer_as_passage_spans,
                                     passage_span_start_log_probs,
                                     passage_span_end_log_probs):
        # Shape: (batch_size, # of answer spans)
        gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
        gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_passage_span_mask = (gold_passage_span_starts != -1).long()
        clamped_gold_passage_span_starts = \
            util.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
        clamped_gold_passage_span_ends = \
            util.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_passage_span_starts = \
            torch.gather(passage_span_start_log_probs, 1, clamped_gold_passage_span_starts)
        log_likelihood_for_passage_span_ends = \
            torch.gather(passage_span_end_log_probs, 1, clamped_gold_passage_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_passage_spans = \
            log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_passage_spans = \
            util.replace_masked_values(log_likelihood_for_passage_spans, gold_passage_span_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)
        return log_marginal_likelihood_for_passage_span
    
    
    def _span_prediction(self, question_passage_tokens, best_span, passage_text, question_text, context):
        (predicted_start, predicted_end)  = tuple(best_span.detach().cpu().numpy())
        answer_tokens = question_passage_tokens[predicted_start:predicted_end + 1]
        spans_text, spans_indices = decode_token_spans([(context, answer_tokens)], passage_text, question_text)
        predicted_answer = spans_text[0]
        return predicted_answer, spans_indices

    
    def _question_span_module(self, passage_vector, question_out, question_mask):
        # Shape: (batch_size, question_length)
        encoded_question_for_span_prediction = \
            torch.cat([question_out,
                       passage_vector.unsqueeze(1).repeat(1, question_out.size(1), 1)], -1)
        question_span_start_logits = \
            self._question_span_start_predictor(encoded_question_for_span_prediction).squeeze(-1)
        # Shape: (batch_size, question_length)
        question_span_end_logits = \
            self._question_span_end_predictor(encoded_question_for_span_prediction).squeeze(-1)
        question_span_start_log_probs = util.masked_log_softmax(question_span_start_logits, question_mask)
        question_span_end_log_probs = util.masked_log_softmax(question_span_end_logits, question_mask)

        # Info about the best question span prediction
        question_span_start_logits = \
            util.replace_masked_values(question_span_start_logits, question_mask, -1e7)
        question_span_end_logits = \
            util.replace_masked_values(question_span_end_logits, question_mask, -1e7)

        # Shape: (batch_size, 2)
        best_question_span = get_best_span(question_span_start_logits, question_span_end_logits)
        return question_span_start_log_probs, question_span_end_log_probs, best_question_span
    
    
    def _question_span_log_likelihood(self, 
                                      answer_as_question_spans, 
                                      question_span_start_log_probs, 
                                      question_span_end_log_probs):
        # Shape: (batch_size, # of answer spans)
        gold_question_span_starts = answer_as_question_spans[:, :, 0]
        gold_question_span_ends = answer_as_question_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_question_span_mask = (gold_question_span_starts != -1).long()
        clamped_gold_question_span_starts = \
            util.replace_masked_values(gold_question_span_starts, gold_question_span_mask, 0)
        clamped_gold_question_span_ends = \
            util.replace_masked_values(gold_question_span_ends, gold_question_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_question_span_starts = \
            torch.gather(question_span_start_log_probs, 1, clamped_gold_question_span_starts)
        log_likelihood_for_question_span_ends = \
            torch.gather(question_span_end_log_probs, 1, clamped_gold_question_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_question_spans = \
            log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_question_spans = \
            util.replace_masked_values(log_likelihood_for_question_spans,
                                       gold_question_span_mask,
                                       -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood_for_question_span = \
            util.logsumexp(log_likelihood_for_question_spans)
        return log_marginal_likelihood_for_question_span
    
    
    def _count_module(self, passage_vector):
        # Shape: (batch_size, 10)
        count_number_logits = self._count_number_predictor(passage_vector)
        count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
        # Info about the best count number prediction
        # Shape: (batch_size,)
        best_count_number = torch.argmax(count_number_log_probs, -1)
        return count_number_log_probs, best_count_number
    

    def _count_log_likelihood(self, answer_as_counts, count_number_log_probs):
        # Count answers are padded with label -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        # Shape: (batch_size, # of count answers)
        gold_count_mask = (answer_as_counts != -1).long()
        # Shape: (batch_size, # of count answers)
        clamped_gold_counts = util.replace_masked_values(answer_as_counts, gold_count_mask, 0)
        log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts)
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_counts = \
            util.replace_masked_values(log_likelihood_for_counts, gold_count_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
        return log_marginal_likelihood_for_count
    
    
    def _count_prediction(self, best_count_number):
        predicted_count = best_count_number.detach().cpu().numpy()
        predicted_answer = str(predicted_count)
        return predicted_answer, predicted_count
    
    
    def _base_arithmetic_module(self, passage_vector, passage_out, number_indices, number_mask):
        if self.number_rep in ['average', 'attention']:
            
            # Shape: (batch_size, # of numbers, # of pieces) 
            number_indices = util.replace_masked_values(number_indices, number_indices != -1, 0).long()
            batch_size = number_indices.shape[0]
            num_numbers = number_indices.shape[1]
            seqlen = passage_out.shape[1]

            # Shape : (batch_size, # of numbers, seqlen)
            mask = torch.zeros((batch_size, num_numbers, seqlen), device=number_indices.device).long().scatter(
                        2, 
                        number_indices, 
                        torch.ones(number_indices.shape, device=number_indices.device).long())
            mask[:,:,0] = 0

            # Shape : (batch_size, # of numbers, seqlen, bert_dim)
            epassage_out = passage_out.unsqueeze(1).repeat(1,num_numbers,1,1)

            # Shape : (batch_size, # of numbers, bert_dim)
            encoded_numbers = self.summary_vector(epassage_out, mask, "numbers")
        else:
            number_indices = number_indices[:,:,0].long()
            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
            encoded_numbers = torch.gather(
                    passage_out,
                    1,
                    clamped_number_indices.unsqueeze(-1).expand(-1, -1, passage_out.size(-1)))
            
        if self.num_special_numbers > 0:
            special_numbers = self.special_embedding(torch.arange(self.num_special_numbers, device=number_indices.device))
            special_numbers = special_numbers.expand(number_indices.shape[0],-1,-1)
            encoded_numbers = torch.cat([special_numbers, encoded_numbers], 1)
            
            mask = torch.ones((number_indices.shape[0],self.num_special_numbers), device=number_indices.device).long()
            number_mask = torch.cat([mask, number_mask], -1)
        # Shape: (batch_size, # of numbers, 2*bert_dim)
        encoded_numbers = torch.cat(
                [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

        # Shape: (batch_size, # of numbers in the passage, 3)
        number_sign_logits = self._number_sign_predictor(encoded_numbers)
        number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

        # Shape: (batch_size, # of numbers in passage).
        best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
        # For padding numbers, the best sign masked as 0 (not included).
        best_signs_for_numbers = util.replace_masked_values(best_signs_for_numbers, number_mask, 0)
        return number_sign_log_probs, best_signs_for_numbers, number_mask
    
    
    def _base_arithmetic_log_likelihood(self,
                                        answer_as_expressions,
                                        number_sign_log_probs,
                                        number_mask, 
                                        answer_as_expressions_extra):
        if self.num_special_numbers > 0:
            answer_as_expressions = torch.cat([answer_as_expressions_extra, answer_as_expressions], -1)
        # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
        # Shape: (batch_size, # of combinations)
        gold_add_sub_mask = (answer_as_expressions.sum(-1) > 0).float()
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        gold_add_sub_signs = answer_as_expressions.transpose(1, 2)
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
        # the log likelihood of the masked positions should be 0
        # so that it will not affect the joint probability
        log_likelihood_for_number_signs = \
            util.replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
        # Shape: (batch_size, # of combinations)
        log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
        # For those padded combinations, we set their log probabilities to be very small negative value
        log_likelihood_for_add_subs = \
            util.replace_masked_values(log_likelihood_for_add_subs, gold_add_sub_mask, -1e7)
        # Shape: (batch_size,)
        log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)
        return log_marginal_likelihood_for_add_sub
    
    
    def _base_arithmetic_prediction(self, original_numbers, number_indices, best_signs_for_numbers):
        sign_remap = {0: 0, 1: 1, 2: -1}
        original_numbers = self.special_numbers + original_numbers
        predicted_signs = [sign_remap[it] for it in best_signs_for_numbers.detach().cpu().numpy()]
        result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
        predicted_answer = str(round(result, 5))
        numbers = []
        for value, sign in zip(original_numbers, predicted_signs):
            numbers.append({'value': value, 'sign': sign})
        if number_indices[-1][0] == -1:
            # There is a dummy 0 number at position -1 added in some cases; we are
            # removing that here.
            numbers.pop()
        return predicted_answer, numbers
    
    
    def _adv_arithmetic_module(self, summary_vector, maxlen, options, options_mask, bert_out, bert_mask):
        # summary_vector: (batch, bert_dim)
        # options : (batch, opnumlen, bert_dim)
        # options_mask : (batch, opnumlen)
        # bert_out : (batch, seqlen, bert_dim)
        # bert_mask: (batch, seqlen)
        
        # summary_vector : (batch, explen, bert_dim)
        summary_vector = summary_vector.unsqueeze(1).expand(-1,maxlen,-1)
        
        # out: (batch, explen, rnn_hdim)
        out, _ = self.rnn(summary_vector)
        out = self.rnndropout(out)
        out = self.Wst(out)

        # alpha : (batch, explen, seqlen)
        alpha = torch.bmm(out, bert_out.transpose(1,2))
        alpha = util.masked_softmax(alpha, bert_mask)
        
        # context : (batch, explen, bert_dim)
        context = util.weighted_sum(bert_out, alpha)
#         context = self.Wcon(context)
        
        # logits : (batch, explen, opnumlen)
        logits = torch.bmm(context, options.transpose(1,2))
        logits =  util.replace_masked_values(logits, options_mask.unsqueeze(1).expand_as(logits), -1e7)
        
        number_mask = options_mask.clone()
        number_mask[:,:self.num_ops] = 0
        op_mask = options_mask.clone()
        op_mask[:,self.num_ops:] = 0
        
        best_expression = beam_search(self.arithmetic_K, logits.softmax(-1),\
                                                number_mask, op_mask, self.END, self.num_ops)
        
        return logits, best_expression[0]
    
    
    def _adv_arithmetic_log_likelihood(self, arithmetic_logits, answer_as_expressions):
        (batch,num_ans,explen) = answer_as_expressions.shape
        opnumlen = arithmetic_logits.shape[-1]
        
        # logits : (batch,#answers,explen,opnumlen)
        arithmetic_logits = arithmetic_logits.unsqueeze(1).expand(batch,num_ans,explen,opnumlen)

        # Shape : (batch,#answers,explen)
        log_likelihood = -self.arith_criterion(arithmetic_logits.permute(0,3,1,2), answer_as_expressions)
        # Shape : (batch,#answer)
        log_likelihood = log_likelihood.sum(-1)
        log_likelihood = util.replace_masked_values(log_likelihood, log_likelihood != 0, -1e7)
        # Shape : (batch,) 
        log_marginal_likelihood_for_arithmetic = util.logsumexp(log_likelihood)
        return log_marginal_likelihood_for_arithmetic
        
        
    def _adv_arithmetic_prediction(self, original_numbers, best_expression):
        with torch.no_grad():
            if best_expression[0] == -100 or (len(best_expression) == self.max_explen):
                return "0", ["None"]
            out = self.ops + original_numbers
            exp_string = [str(out[idx]) for idx in best_expression]
            return evaluate_postfix(exp_string), exp_string
    
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        (exact_match, f1_score), scores_per_answer_type_and_head, \
        scores_per_answer_type, scores_per_head = self._drop_metrics.get_metric(reset)
        metrics = {'em': exact_match, 'f1': f1_score}

        for answer_type, type_scores_per_head in scores_per_answer_type_and_head.items():
            for head, (answer_type_head_exact_match, answer_type_head_f1_score) in type_scores_per_head.items():
                if 'multi' in head and 'span' in answer_type:
                    metrics[f'em_{answer_type}_{head}'] = answer_type_head_exact_match
                    metrics[f'f1_{answer_type}_{head}'] = answer_type_head_f1_score
                else:
                    metrics[f'_em_{answer_type}_{head}'] = answer_type_head_exact_match
                    metrics[f'_f1_{answer_type}_{head}'] = answer_type_head_f1_score
        
        for answer_type, (type_exact_match, type_f1_score) in scores_per_answer_type.items():
            if 'span' in answer_type:
                metrics[f'em_{answer_type}'] = type_exact_match
                metrics[f'f1_{answer_type}'] = type_f1_score
            else:
                metrics[f'_em_{answer_type}'] = type_exact_match
                metrics[f'_f1_{answer_type}'] = type_f1_score

        for head, (head_exact_match, head_f1_score) in scores_per_head.items():
            if 'multi' in head:
                metrics[f'em_{head}'] = head_exact_match
                metrics[f'f1_{head}'] = head_f1_score
            else:
                metrics[f'_em_{head}'] = head_exact_match
                metrics[f'_f1_{head}'] = head_f1_score
        
        return metrics
