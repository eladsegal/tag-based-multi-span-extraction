"""
Copied from allennlp's conditional_random_field.py, with some changes
"""

from typing import List, Tuple, Dict, Union

import torch

from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util

VITERBI_DECODING = Tuple[List[int], float]  # a list of tags, and a viterbi score


def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : ``Dict[int, str]``, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    ``List[Tuple[int, int]]``
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings ``from_tag`` and ``to_tag`` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : ``str``, required
        The tag that the transition originates from. For example, if the
        label is ``I-PER``, the ``from_tag`` is ``I``.
    from_entity: ``str``, required
        The entity corresponding to the ``from_tag``. For example, if the
        label is ``I-PER``, the ``from_entity`` is ``PER``.
    to_tag : ``str``, required
        The tag that the transition leads to. For example, if the
        label is ``I-PER``, the ``to_tag`` is ``I``.
    to_entity: ``str``, required
        The entity corresponding to the ``to_tag``. For example, if the
        label is ``I-PER``, the ``to_entity`` is ``PER``.

    Returns
    -------
    ``bool``
        Whether the transition is allowed under the given ``constraint_type``.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I") and to_tag in ("I", "L") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag in ("B", "M") and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")

def viterbi_tags(
    logits: torch.Tensor, 
    transitions: torch.Tensor, 
    constraint_mask: torch.Tensor, 
    mask: torch.Tensor = None,
    top_k: int = None
) -> Union[List[VITERBI_DECODING], List[List[VITERBI_DECODING]]]:
    """
    Uses viterbi algorithm to find most likely tags for the given inputs.
    If constraints are applied, disallows all other transitions.

    Returns a list of results, of the same size as the batch (one result per batch member)
    Each result is a List of length top_k, containing the top K viterbi decodings
    Each decoding is a tuple  (tag_sequence, viterbi_score)

    For backwards compatibility, if top_k is None, then instead returns a flat list of
    tag sequences (the top tag sequence for each batch item).
    """
    _transitions = transitions # because transitions is overwritten
    if mask is None:
        mask = torch.ones(*logits.shape[:2], dtype=torch.long, device=logits.device)

    if top_k is None:
        top_k = 1
        flatten_output = True
    else:
        flatten_output = False

    batch_size, max_seq_length, num_tags = logits.size()

    # Get the tensors out of the variables
    logits, mask = logits.data, mask.data

    # Augment transitions matrix with start and end transitions
    start_tag = num_tags
    end_tag = num_tags + 1
    transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)

    # Apply transition constraints
    constrained_transitions = _transitions * constraint_mask[
        :num_tags, :num_tags
    ] + -10000.0 * (1 - constraint_mask[:num_tags, :num_tags])
    transitions[:num_tags, :num_tags] = constrained_transitions.data

    transitions[start_tag, :num_tags] = -10000.0 * (
        1 - constraint_mask[start_tag, :num_tags].detach()
    )
    transitions[:num_tags, end_tag] = -10000.0 * (
        1 - constraint_mask[:num_tags, end_tag].detach()
    )

    best_paths = []
    # Pad the max sequence length by 2 to account for start_tag + end_tag.
    tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

    for prediction, prediction_mask in zip(logits, mask):
        sequence_length = torch.sum(prediction_mask)
        mask_indices = prediction_mask.nonzero().squeeze()
        prediction = torch.index_select(prediction, 0, mask_indices)

        # Start with everything totally unlikely
        tag_sequence.fill_(-10000.0)
        # At timestep 0 we must have the START_TAG
        tag_sequence[0, start_tag] = 0.0
        # At steps 1, ..., sequence_length we just use the incoming prediction
        tag_sequence[1 : (sequence_length + 1), :num_tags] = prediction
        # And at the last timestep we must have the END_TAG
        tag_sequence[sequence_length + 1, end_tag] = 0.0

        # We pass the tags and the transitions to ``viterbi_decode``.
        viterbi_paths, viterbi_scores = util.viterbi_decode(
            tag_sequence=tag_sequence[: (sequence_length + 2)],
            transition_matrix=transitions,
            top_k=top_k,
        )
        top_k_paths = []
        for viterbi_path in viterbi_paths:
            # Get rid of START and END sentinels and append.
            viterbi_path = viterbi_path[1:-1]
            if batch_size > 1 and len(viterbi_path) < max_seq_length:
                viterbi_path += [0] * (max_seq_length - len(viterbi_path))
            top_k_paths.append(viterbi_path)
        best_paths.append(top_k_paths)

    if flatten_output:
        return [top_k_paths[0] for top_k_paths in best_paths]

    return best_paths
