from typing import Any, Dict, List, Optional, Union
import logging

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import FeedForward
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from transformers import AutoModel

from src.modules.heads.head import Head
from src.training.metrics.custom_em_and_f1 import CustomEmAndF1

logger = logging.getLogger(__name__)

@Model.register('multi_head')
class MultiHeadModel(Model):
    def __init__(self, 
                vocab: Vocabulary,
                pretrained_model: str,
                heads: Dict[str, Head],
                dataset_name,
                head_predictor: Optional[FeedForward] = None,
                passage_summary_vector_module: Optional[FeedForward] = None,
                question_summary_vector_module: Optional[FeedForward] = None,
                training_evaluation: bool = True,
                output_all_answers: bool = False,
                initializer: InitializerApplicator = InitializerApplicator(),
                regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._pretrained_model = pretrained_model
        self._transformers_model = AutoModel.from_pretrained(pretrained_model)
        
        self._heads = torch.nn.ModuleDict(heads)
        self._head_predictor = head_predictor
        self._passage_summary_vector_module = passage_summary_vector_module
        self._question_summary_vector_module = question_summary_vector_module

        self._training_evaluation = training_evaluation

        self._output_all_answers = output_all_answers

        self._metrics = CustomEmAndF1(dataset_name)

        initializer(self)

    def heads_indices(self):
        return list(self._heads.keys())

    def summary_vector(self, encoding, mask, in_type='passage'):
        """
        In NABERT (and in NAQANET), a 'summary_vector' is created for some entities, such as the
        passage or the question. This vector is created as a weighted sum of the elements of the
        entity, e.g. the passage summary vector is a weighted sum of the passage tokens.

        The specific weighting for every entity type is a learned.

        Parameters
        ----------
        encoding : Pretrained-model's output layer
        mask : a Tensor with 1s only at the positions relevant to ``in_type``
        in_type : the entity we want to summarize, e.g the passage

        Returns
        -------
        The summary vector according to ``in_type``.
        """
        if in_type == 'passage':
            # Shape: (batch_size, seqlen)
            alpha = self._passage_summary_vector_module(encoding).squeeze()
        elif in_type == 'question':
            # Shape: (batch_size, seqlen)
            alpha = self._question_summary_vector_module(encoding).squeeze()
        else:
            # Shape: (batch_size, #num of numbers, seqlen)
            alpha = torch.zeros(encoding.shape[:-1], device=encoding.device)
        # Shape: (batch_size, seqlen) 
        # (batch_size, #num of numbers, seqlen) for numbers
        alpha = masked_softmax(alpha, mask)
        # Shape: (batch_size, out)
        # (batch_size, #num of numbers, out) for numbers
        h = util.weighted_sum(encoding, alpha)
        return h
        
    def forward(self,  # type: ignore
                question_passage_tokens: torch.LongTensor,
                question_passage_token_type_ids: torch.LongTensor,
                question_passage_special_tokens_mask: torch.LongTensor,
                question_passage_pad_mask: torch.LongTensor,
                first_wordpiece_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                wordpiece_indices: torch.LongTensor = None,
                number_indices: torch.LongTensor = None,
                answer_as_expressions: torch.LongTensor = None,
                answer_as_expressions_extra: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                answer_as_text_to_disjoint_bios: torch.LongTensor = None,
                answer_as_list_of_bios: torch.LongTensor = None,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                span_bio_labels: torch.LongTensor = None,
                is_bio_mask: torch.LongTensor = None) -> Dict[str, Any]:
        # pylint: disable=arguments-differ
        question_passage_special_tokens_mask = (1 - question_passage_special_tokens_mask)
        
        batch_size = question_passage_tokens.shape[0]
        head_count = len(self._heads)

        # TODO: (not important) Create a new field that is converted to Dict[str, torch.LongTensor]
        gold_answer_representations = {
            'answer_as_expressions': answer_as_expressions,
            'answer_as_expressions_extra': answer_as_expressions_extra,
            'answer_as_passage_spans': answer_as_passage_spans,
            'answer_as_question_spans': answer_as_question_spans,
            'answer_as_counts': answer_as_counts,
            'answer_as_text_to_disjoint_bios': answer_as_text_to_disjoint_bios,
            'answer_as_list_of_bios': answer_as_list_of_bios,
            'span_bio_labels': span_bio_labels
        }

        has_answer = False
        for answer_representation in gold_answer_representations.values():
            if answer_representation is not None:
                has_answer = True
                break

        # Shape: (batch_size, seqlen)
        passage_mask = question_passage_token_type_ids * question_passage_pad_mask * question_passage_special_tokens_mask
        # Shape: (batch_size, seqlen)
        question_mask = (1 - question_passage_token_type_ids) * question_passage_pad_mask * question_passage_special_tokens_mask
        question_and_passage_mask = question_mask | passage_mask

        # Shape: (batch_size, seqlen, bert_dim)
        token_representations = self._transformers_model(question_passage_tokens, 
                                             token_type_ids=(question_passage_token_type_ids 
                                                             if not self._pretrained_model.startswith('roberta-') 
                                                             else None), 
                                             attention_mask=question_passage_pad_mask)[0]
        
        if self._passage_summary_vector_module is not None:
            # Shape: (batch_size, bert_dim)
            passage_summary_vector = self.summary_vector(token_representations, passage_mask, 'passage')
        else:
            passage_summary_vector = None

        if self._question_summary_vector_module is not None:
            # Shape: (batch_size, bert_dim)
            question_summary_vector = self.summary_vector(token_representations, question_mask, 'question')
        else:
            question_summary_vector = None

        if head_count > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._head_predictor(torch.cat([passage_summary_vector, question_summary_vector], -1))
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            top_answer_abilities = torch.argsort(answer_ability_log_probs, descending=True)
        else:
            top_answer_abilities = torch.zeros(batch_size, 1, dtype=torch.int)

        kwargs = {
            'token_representations': token_representations,
            'passage_summary_vector': passage_summary_vector,
            'question_summary_vector': question_summary_vector,
            'gold_answer_representations': gold_answer_representations,
            'question_and_passage_mask': question_and_passage_mask,
            'first_wordpiece_mask': first_wordpiece_mask,
            'is_bio_mask': is_bio_mask,
            'wordpiece_indices': wordpiece_indices,
            'number_indices': number_indices,
            'passage_mask': passage_mask,
            'question_mask': question_mask,
            'question_passage_special_tokens_mask': question_passage_special_tokens_mask
        }

        head_outputs = {}
        for head_name, head in self._heads.items():
            head_outputs[head_name] = head(**kwargs)
                        
        output_dict = {}
        # If answer is given, compute the loss.
        if has_answer:
            log_marginal_likelihood_list = []
            for head_name, head in self._heads.items():
                log_marginal_likelihood = head.gold_log_marginal_likelihood(**kwargs, **head_outputs[head_name])
                log_marginal_likelihood_list.append(log_marginal_likelihood)
            
            if head_count > 1:
                # Add the ability probabilities if there is more than one ability
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
        
            output_dict['loss'] = -1 * marginal_log_likelihood.mean()

        with torch.no_grad():
            # Compute the metrics and add fields to the output
            if metadata is not None and self._training_evaluation:
                if not self.training:
                    output_dict['passage_id'] = []
                    output_dict['query_id'] = []
                    output_dict['answer'] = []
                    output_dict['predicted_ability'] = []
                    output_dict['maximizing_ground_truth'] = []
                    output_dict['em'] = []
                    output_dict['f1'] = []
                    output_dict['max_passage_length'] = []
                    if self._output_all_answers:
                        output_dict['all_answers'] = []

                i = 0
                no_fallback = False
                ordered_lookup_index = 0
                while i < batch_size:
                    predicting_head_index = top_answer_abilities[i][ordered_lookup_index].item()
                    predicting_head_name = self.heads_indices()[predicting_head_index]
                    predicting_head = self._heads[predicting_head_name]

                    # construct the arguments to be used for a batch instance prediction
                    instance_kwargs = {
                        'q_text': metadata[i]['original_question'],
                        'p_text': metadata[i]['original_passage'],
                        'qp_tokens': metadata[i]['question_passage_tokens'],
                        'question_passage_wordpieces': metadata[i]['question_passage_wordpieces'],
                        'original_numbers': metadata[i]['original_numbers'] if 'original_numbers' in metadata[i] else None,
                    }

                    # keys that cannot be passed because 
                    # they are not batch-based in their first level or None
                    unpassable_keys = ['gold_answer_representations']

                    for key, value in instance_kwargs.items():
                        if value is None:
                            unpassable_keys.append(key)
                    for key in unpassable_keys:
                        if key in instance_kwargs:
                            del instance_kwargs[key]

                    for key, value in kwargs.items():
                        if value is not None and key not in unpassable_keys:
                            instance_kwargs[key] = value[i]
                    for key, value in head_outputs[predicting_head_name].items():
                        if key not in unpassable_keys:
                            instance_kwargs[key] = value[i]


                    # get prediction for an instance in the batch
                    answer_json = predicting_head.decode_answer(**instance_kwargs)

                    if len(answer_json['value']) != 0 or no_fallback:
                        # for the next in the batch
                        ordered_lookup_index = 0
                        no_fallback = False
                    else:
                        if not self.training:
                            logger.info("Answer was empty for head: %s, query_id: %s", predicting_head_name, metadata[i]['question_id'])
                        ordered_lookup_index += 1
                        if ordered_lookup_index == head_count:
                            no_fallback = True
                            ordered_lookup_index = 0
                        continue

                    maximizing_ground_truth = None
                    em, f1 = None, None
                    answer_annotations = metadata[i].get('answer_annotations', [])
                    if answer_annotations:
                        (em, f1), maximizing_ground_truth = self._metrics.call(answer_json['value'], answer_annotations, predicting_head_name)

                    if not self.training:
                        output_dict['passage_id'].append(metadata[i]['passage_id'])
                        output_dict['query_id'].append(metadata[i]['question_id'])
                        output_dict['answer'].append(answer_json)
                        output_dict['predicted_ability'].append(predicting_head_name)
                        output_dict['maximizing_ground_truth'].append(maximizing_ground_truth)
                        output_dict['em'].append(em)
                        output_dict['f1'].append(f1)
                        output_dict['max_passage_length'].append(metadata[i]['max_passage_length'])
                        
                        if self._output_all_answers:
                            answers_dict = {}
                            output_dict['all_answers'].append(answers_dict)
                            for j in range(len(self._heads)):
                                predicting_head_index = top_answer_abilities[i][j].item()
                                predicting_head_name = self.heads_indices()[predicting_head_index]
                                predicting_head = self._heads[predicting_head_name]

                                # construct the arguments to be used for a batch instance prediction
                                instance_kwargs = {
                                    'q_text': metadata[i]['original_question'],
                                    'p_text': metadata[i]['original_passage'],
                                    'qp_tokens': metadata[i]['question_passage_tokens'],
                                    'question_passage_wordpieces': metadata[i]['question_passage_wordpieces'],
                                    'original_numbers': metadata[i]['original_numbers'] if 'original_numbers' in metadata[i] else None,
                                }

                                # keys that cannot be passed because 
                                # they are not batch-based in their first level or None
                                unpassable_keys = ['gold_answer_representations']

                                for key, value in instance_kwargs.items():
                                    if value is None:
                                        unpassable_keys.append(key)
                                for key in unpassable_keys:
                                    if key in instance_kwargs:
                                        del instance_kwargs[key]

                                for key, value in kwargs.items():
                                    if value is not None and key not in unpassable_keys:
                                        instance_kwargs[key] = value[i]
                                for key, value in head_outputs[predicting_head_name].items():
                                    if key not in unpassable_keys:
                                        instance_kwargs[key] = value[i]

                                # get prediction for an instance in the batch
                                answer_json = predicting_head.decode_answer(**instance_kwargs)
                                answer_json['probability'] = torch.nn.functional.softmax(answer_ability_logits, -1)[i][predicting_head_index].item()
                                answers_dict[predicting_head_name] = answer_json

                    i += 1

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        (exact_match, f1_score), scores_per_answer_type_and_head, \
        scores_per_answer_type, scores_per_head = self._metrics.get_metric(reset)
        metrics = {'em': exact_match, 'f1': f1_score}

        for answer_type, type_scores_per_head in scores_per_answer_type_and_head.items():
            for head, (answer_type_head_exact_match, answer_type_head_f1_score, type_head_count) in type_scores_per_head.items():
                if 'multi' in head and 'span' in answer_type:
                    metrics[f'em_{answer_type}_{head}'] = answer_type_head_exact_match
                    metrics[f'f1_{answer_type}_{head}'] = answer_type_head_f1_score
                else:
                    metrics[f'_em_{answer_type}_{head}'] = answer_type_head_exact_match
                    metrics[f'_f1_{answer_type}_{head}'] = answer_type_head_f1_score
                metrics[f'_counter_{answer_type}_{head}'] = type_head_count
        
        for answer_type, (type_exact_match, type_f1_score, type_count) in scores_per_answer_type.items():
            if 'span' in answer_type:
                metrics[f'em_{answer_type}'] = type_exact_match
                metrics[f'f1_{answer_type}'] = type_f1_score
            else:
                metrics[f'_em_{answer_type}'] = type_exact_match
                metrics[f'_f1_{answer_type}'] = type_f1_score
            metrics[f'_counter_{answer_type}'] = type_count

        for head, (head_exact_match, head_f1_score, head_count) in scores_per_head.items():
            if 'multi' in head:
                metrics[f'em_{head}'] = head_exact_match
                metrics[f'f1_{head}'] = head_f1_score
            else:
                metrics[f'_em_{head}'] = head_exact_match
                metrics[f'_f1_{head}'] = head_f1_score
            metrics[f'_counter_{head}'] = head_count

        if '_counter_span' in metrics and '_counter_spans' in metrics:
            total = metrics['_counter_span'] + metrics['_counter_spans']
            span_percentage = metrics['_counter_span'] / total
            spans_percentage = metrics['_counter_spans'] / total
            em_all_spans = (span_percentage * metrics['em_span']) + (spans_percentage * metrics['em_spans'])
            f1_all_spans = (span_percentage * metrics['f1_span']) + (spans_percentage * metrics['f1_spans'])

            metrics['em_all_spans'] = em_all_spans
            metrics['f1_all_spans'] = f1_all_spans
        
        return metrics
