###############################
# Testing
###############################
import sys,os
sys.path.append(os.getcwd())
import src.top_k_viterbi

from torch.autograd import Variable
from tqdm import tqdm
import random
import numpy as np


def test_greedy():
    # Test Viterbi decoding is equal to greedy decoding with no pairwise potentials.
    sequence_logits = Variable(torch.rand([5, 9]))
    transition_matrix = torch.zeros([9, 9])
    indices, _ = viterbi_decode(sequence_logits.data, transition_matrix)
    _, argmax_indices = torch.max(sequence_logits, 1)
    assert indices[0] == argmax_indices.data.squeeze().tolist()


test_greedy()
print('PASSED TEST GREEDY')


def test_inf():
    # Test that pairwise potentials effect the sequence correctly and that
    # viterbi_decode can handle -inf values.
    sequence_logits = torch.FloatTensor([[0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4],
                                         [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4]])
    # The same tags shouldn't appear sequentially.
    transition_matrix = torch.zeros([5, 5])
    for i in range(5):
        transition_matrix[i, i] = float("-inf")
    indices, _ = viterbi_decode(sequence_logits, transition_matrix)
    assert indices[0] == [3, 4, 3, 4, 3, 4]


test_inf()
print('PASSED TEST INF')


def test_ties():
    # Test that unbalanced pairwise potentials break ties
    # between paths with equal unary potentials.
    sequence_logits = torch.FloatTensor([[0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4],
                                         [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4]])
    # The 5th tag has a penalty for appearing sequentially
    # or for transitioning to the 4th tag, making the best
    # path uniquely to take the 4th tag only.
    transition_matrix = torch.zeros([5, 5])
    transition_matrix[4, 4] = -10
    transition_matrix[4, 3] = -10
    indices, _ = viterbi_decode(sequence_logits, transition_matrix)
    assert indices[0] == [3, 3, 3, 3, 3, 3]


test_ties()
print('PASSED TEST TIES')


def test_transitions():
    sequence_logits = torch.FloatTensor([[1, 0, 0, 4], [1, 0, 6, 2], [0, 3, 0, 4]])
    # Best path would normally be [3, 2, 3] but we add a
    # potential from 2 -> 1, making [3, 2, 1] the best path.
    transition_matrix = torch.zeros([4, 4])
    transition_matrix[0, 0] = 1
    transition_matrix[2, 1] = 5
    indices, value = viterbi_decode(sequence_logits, transition_matrix)
    assert indices[0] == [3, 2, 1]
    assert value[0] == 18


test_transitions()
print('PASSED TEST TRANSITIONS')


# Use the brute decoding as truth
def brute_decode(tag_sequence: torch.Tensor, transition_matrix: torch.Tensor, top_k: int=5):
    """
    Top-k decoder that uses brute search instead of the Viterbi Decode dynamic programing algorithm
    """
    # Create all possible sequences
    sequence_length, num_tags = list(tag_sequence.size())
    sequences = [[]]
    for i in range(len(tag_sequence)):
        new_sequences = []
        for j in range(len(tag_sequence[i])):
            for sequence in sequences:
                new_sequences.append(sequence[:] + [j])
        sequences = new_sequences

    # Score
    scored_sequences = []
    for sequence in sequences:
        emission_score = sum([tag_sequence[i, j] for i, j in enumerate(sequence)])
        transition_score = sum(
            [transition_matrix[sequence[i - 1], sequence[i]] for i in range(1, len(sequence))])
        score = emission_score + transition_score
        scored_sequences.append((score, sequence))

    # Get the top k scores / paths
    top_k_sequences = sorted(scored_sequences, key=lambda r: r[0], reverse=True)[:top_k]
    scores, paths = zip(*top_k_sequences)

    return paths, scores


def test_brute():
    # Run 100 randomly generated parameters and compare the outputs.
    for i in tqdm(range(100)):
        num_tags = random.randint(1, 5)
        seq_len = random.randint(1, 5)
        k = random.randint(1, 5)
        sequence_logits = torch.rand([seq_len, num_tags])
        transition_matrix = torch.rand([num_tags, num_tags])
        viterbi_paths_v1, viterbi_scores_v1 = viterbi_decode(
            sequence_logits, transition_matrix, top_k=k)
        viterbi_path_brute, viterbi_score_brute = brute_decode(
            sequence_logits, transition_matrix, top_k=k)
        np.testing.assert_almost_equal(
            list(viterbi_score_brute), viterbi_scores_v1.tolist(), decimal=3)


test_brute()
print('PASSED TEST BRUTE')