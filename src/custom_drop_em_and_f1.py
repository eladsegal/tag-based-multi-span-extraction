from typing import Tuple, List, Union

from collections import defaultdict

from overrides import overrides

from allennlp.tools.drop_eval import (get_metrics as drop_em_and_f1,
                                      answer_json_to_strings)
from allennlp.training.metrics.metric import Metric


@Metric.register("custom_drop")
class CustomDropEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._answer_type_em = defaultdict(float)
        self._answer_type_f1 = defaultdict(float)
        self._answer_type_count = defaultdict(int)

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

    def call(self, prediction: Union[str, List], ground_truths: List) -> Union[str, List]:
        # If you wanted to split this out by answer type, you could look at [1] here and group by
        # that, instead of only keeping [0].
        ground_truth_answer_strings, ground_truth_answer_types = list(zip(*[answer_json_to_strings(annotation) for annotation in ground_truths]))
        (exact_match, f1_score), maximizing_ground_truth_index = CustomDropEmAndF1.metric_max_over_ground_truths(
                drop_em_and_f1,
                prediction,
                ground_truth_answer_strings
        )
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

        # Have to select one ground truth, so might as well be the maximizing one
        answer_type = ground_truth_answer_types[maximizing_ground_truth_index]
        self._answer_type_em[answer_type] += exact_match
        self._answer_type_f1[answer_type] += f1_score
        self._answer_type_count[answer_type] += 1

        return (exact_match, f1_score), ground_truth_answer_strings[maximizing_ground_truth_index]

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        
        scores_per_answer_type = {}
        for answer_type, count in self._answer_type_count.items():
            exact_match = self._answer_type_em[answer_type] / count if count > 0 else 0
            f1_score = self._answer_type_f1[answer_type] / count if count > 0 else 0
            scores_per_answer_type[answer_type] = exact_match, f1_score
        
        if reset:
            self.reset()
        return exact_match, f1_score

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._answer_type_em = defaultdict(float)
        self._answer_type_f1 = defaultdict(float)
        self._answer_type_count = defaultdict(int)
    
    def __str__(self):
        return f"CustomDropEmAndF1(em={self._total_em}, f1={self._total_f1}, answer_type_em={self._answer_type_em}, answer_type_f1={self._answer_type_f1})"


    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        """
        Modified from squad_eval.py in allennlp        
        
        Returns
        -------
        Maximum metric value and the matching index of ground truth
        """
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        maximizing_index = max(range(len(scores_for_ground_truths)), key=lambda i: scores_for_ground_truths[i])
        return scores_for_ground_truths[maximizing_index], maximizing_index
