from typing import Dict, Any

import torch

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules import FeedForward

from src.modules.wordpiece_unifiers.wordpiece_unifier import WordpieceUnifier

@WordpieceUnifier.register('independent')
class IndependentWordpieceUnifier(WordpieceUnifier):
    def __init__(self,
                 output_layer: FeedForward) -> None:
        super().__init__()
        self._output_layer = output_layer

    def forward(self,
                wordpiece_representations: torch.Tensor,
                mask: torch.Tensor,
                **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        logits = self._output_layer(wordpiece_representations)
        alpha = masked_softmax(logits, mask)
        unified_representation = weighted_sum(wordpiece_representations, 
                                              alpha)
        return unified_representation
