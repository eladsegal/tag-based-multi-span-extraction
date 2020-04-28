from typing import Tuple, List, Union, Dict, Any

from collections import defaultdict

from overrides import overrides

from allennlp.tools.drop_eval import (get_metrics as drop_em_and_f1,
                                      answer_json_to_strings)
from allennlp.training.metrics.metric import Metric


@Metric.register("custom_em_and_f1")
class CustomEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self, dataset_name='drop') -> None:
        self._dataset_name = dataset_name

        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._answer_type_head_em = defaultdict(lambda: defaultdict(float))
        self._answer_type_head_f1 = defaultdict(lambda: defaultdict(float))
        self._answer_type_head_count = defaultdict(lambda: defaultdict(int))

    @overrides
    def __call__(self, prediction: Union[str, List], ground_truths: List):  # type: ignore
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List``
            All the ground truth answer annotations.
        """
        self.call(prediction, ground_truths)

    def call(self, prediction: Union[str, List], ground_truths: List, predicting_head: str = None) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        ground_truth_answer_strings, ground_truth_answer_types = list(zip(*[self.annotation_to_answer_and_type(annotation) for annotation in ground_truths]))
        (exact_match, f1_score), maximizing_ground_truth_index = self.metric_max_over_ground_truths(
                drop_em_and_f1,
                prediction,
                ground_truth_answer_strings
        )
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

        # Best answer type is selected, just as in drop_eval
        answer_type = ground_truth_answer_types[maximizing_ground_truth_index]
        self._answer_type_head_em[answer_type][predicting_head] += exact_match
        self._answer_type_head_f1[answer_type][predicting_head] += f1_score
        self._answer_type_head_count[answer_type][predicting_head] += 1

        return (exact_match, f1_score), ground_truths[maximizing_ground_truth_index]

    def annotation_to_answer_and_type(self, annotation) -> Tuple[Tuple[str, ...], str]:
        if self._dataset_name == 'quoref':
            return answer_json_to_strings(annotation) # because we use a dropified version
        else:
            return answer_json_to_strings(annotation)
        

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[Tuple[float, float], Dict[str, Dict[str, Tuple[float, float, int]]], Dict[str, Tuple[float, float, int]], Dict[str, Tuple[float, float, int]]]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        
        scores_per_answer_type_and_head = defaultdict(lambda: {})
        scores_per_answer_type = {}
        scores_per_head = {}

        em_per_head = defaultdict(float)
        f1_per_head = defaultdict(float)
        count_per_head = defaultdict(int)

        for answer_type, head_count in self._answer_type_head_count.items():
            type_count = 0
            type_em = 0.0
            type_f1 = 0.0

            for head, count in head_count.items():
                type_count += count
                type_em += self._answer_type_head_em[answer_type][head]
                type_f1 += self._answer_type_head_f1[answer_type][head]

                em_per_head[head] += self._answer_type_head_em[answer_type][head]
                f1_per_head[head] += self._answer_type_head_f1[answer_type][head]
                count_per_head[head] += count

                type_head_exact_match = self._answer_type_head_em[answer_type][head] / count
                type_head_f1_score = self._answer_type_head_f1[answer_type][head] / count
                scores_per_answer_type_and_head[answer_type][head] = type_head_exact_match, type_head_f1_score, count
            
            scores_per_answer_type[answer_type] = type_em / type_count, type_f1 / type_count, type_count

        for head, count in count_per_head.items():
            scores_per_head[head] = em_per_head[head] / count, f1_per_head[head] / count, count
        
        if reset:
            self.reset()
        return (exact_match, f1_score), scores_per_answer_type_and_head, scores_per_answer_type, scores_per_head

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._answer_type_head_em = defaultdict(lambda: defaultdict(float))
        self._answer_type_head_f1 = defaultdict(lambda: defaultdict(float))
        self._answer_type_head_count = defaultdict(lambda: defaultdict(int))
    
    def __str__(self):
        return f"CustomEmAndF1(em={self._total_em}, f1={self._total_f1}, _answer_type_head_em={self._answer_type_head_em}, _answer_type_head_count={self._answer_type_head_count})"


    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        """
        Modified from squad_eval.py in allennlp, changed to return maximizing index and match drop_eval
        
        Returns
        -------
        Maximum metric value and the matching index of ground truth
        """
        max_em_score = 0.0
        max_f1_score = 0.0
        maximizing_index = -1
        for i, ground_truth in enumerate(ground_truths):
            em_score, f1_score = metric_fn(prediction, ground_truth)
            if ground_truth[0].strip() != "":
                max_em_score = max(max_em_score, em_score)
                max_f1_score = max(max_f1_score, f1_score)
                if max_em_score == em_score and max_f1_score == f1_score:
                    maximizing_index = i

        return (max_em_score, max_f1_score), maximizing_index
