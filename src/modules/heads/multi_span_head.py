from typing import Tuple, Dict, Any, List, Union, Optional

import numpy as np 
import torch

from itertools import product
from collections import OrderedDict

from allennlp.nn.util import replace_masked_values, logsumexp, viterbi_decode
from allennlp.data.tokenizers import Token
from allennlp.modules import FeedForward

from src.modules.heads.head import Head
from src.modules.utils.viterbi_decoding import allowed_transitions, viterbi_tags
from src.modules.utils.decoding_utils import decode_token_spans, get_token_context

@Head.register('multi_span_head')
class MultiSpanHead(Head):
    def __init__(self,
                 output_layer: FeedForward,
                 ignore_question: bool,
                 prediction_method: str,
                 decoding_style: str,
                 training_style: str,
                 labels: Dict[str, int],
                 generation_top_k: int = 0,
                 unique_decoding: bool = True,
                 substring_unique_decoding: bool = True) -> None:
        super().__init__()
        self._output_layer = output_layer
        self._ignore_question = ignore_question
        self._generation_top_k = generation_top_k
        self._unique_decoding = unique_decoding
        self._substring_unique_decoding = substring_unique_decoding
        self._prediction_method = prediction_method
        self._decoding_style = decoding_style
        self._training_style = training_style
        self._labels = labels

        assert(labels['O'] == 0) # must have O as 0 as there are assumptions about it
        self._labels_scheme = ''.join(sorted(labels.keys()))
        if self._labels_scheme == 'BILOU':
            self._labels_scheme = 'BIOUL'
            self._span_start_label = self._labels['U']
        else:
            self._span_start_label = self._labels['B'] if self._labels_scheme == 'BIO' else self._labels['I']

        if self._prediction_method == 'viterbi':
            num_tags = len(labels)
            self._transitions = torch.ones(num_tags, num_tags)
            self._constraint_mask = constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.0)

            if self._labels_scheme == 'BIO' or self._labels_scheme == 'BIOUL':
                constraints = allowed_transitions(self._labels_scheme, {value: key for key, value in labels.items()})
            else:
                constraints = list(product(range(num_tags), range(num_tags)))
                constraints += [(num_tags, i) for i in range(num_tags)]
                constraints += [(i, num_tags + 1) for i in range(num_tags)]

            for i, j in constraints:
                constraint_mask[i, j] = 1.0

    def _get_mask(self,
                  question_and_passage_mask: torch.LongTensor,
                  passage_mask: torch.LongTensor,
                  first_wordpiece_mask: torch.LongTensor) -> torch.Tensor:
        if self._ignore_question:
            mask = passage_mask
        else:
            mask = question_and_passage_mask

        if self._decoding_style == "single_word_representation":
            mask *= first_wordpiece_mask

        return mask

    def forward(self,
                token_representations: torch.LongTensor,
                question_and_passage_mask: torch.LongTensor,
                passage_mask: torch.LongTensor,
                first_wordpiece_mask: torch.LongTensor,
                wordpiece_indices: torch.LongTensor,
                **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        mask = self._get_mask(question_and_passage_mask, passage_mask, first_wordpiece_mask)

        logits = self._output_layer(token_representations)

        log_probs = replace_masked_values(torch.nn.functional.log_softmax(logits, dim=-1), mask.unsqueeze(-1), 0.0)

        output_dict = {
            'log_probs': log_probs
        }
        return output_dict

    def _get_wordpiece_unified_representation(self, token_representations, wordpiece_indices):
        wordpiece_mask = (wordpiece_indices != -1).long()
        wordpiece_indices = replace_masked_values(wordpiece_indices, wordpiece_mask, 0).long()
        batch_size = wordpiece_indices.shape[0]
        num_wordpieces = wordpiece_indices.shape[1]
        seq_len = token_representations.shape[-1]

        mask = torch.zeros((batch_size, num_wordpieces, seq_len), device=wordpiece_indices.device).long().scatter(
                    2, 
                    wordpiece_indices, 
                    torch.ones(wordpiece_indices.shape, device=wordpiece_indices.device).long())
        mask[:,:,0] = 0

        expanded_token_representations = token_representations.unsqueeze(1).repeat(1, num_wordpieces, 1, 1)

        #clamped_wordpiece_indices = 

    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 log_probs: torch.LongTensor,
                                 question_and_passage_mask: torch.LongTensor,
                                 passage_mask: torch.LongTensor,
                                 first_wordpiece_mask: torch.LongTensor,
                                 is_bio_mask: torch.LongTensor,
                                 **kwargs: Any):
        mask = self._get_mask(question_and_passage_mask, passage_mask, first_wordpiece_mask)

        gold_bio_seqs = self._get_gold_answer(gold_answer_representations, log_probs, mask)
        if self._training_style == 'soft_em':
            log_marginal_likelihood = self._marginal_likelihood(gold_bio_seqs, log_probs)
        elif self._training_style == 'hard_em':
            log_marginal_likelihood = self._get_most_likely_likelihood(gold_bio_seqs, log_probs)
        else:
            raise Exception("Illegal training_style")

        # For questions without spans, we set their log likelihood to be very small negative value
        log_marginal_likelihood = replace_masked_values(log_marginal_likelihood, is_bio_mask, -1e7)

        return log_marginal_likelihood

    def decode_answer(self,
                      log_probs: torch.LongTensor,
                      qp_tokens: List[Token],
                      p_text: str,
                      q_text: str,
                      question_passage_wordpieces: List[List[int]],
                      question_and_passage_mask: torch.LongTensor,
                      passage_mask: torch.LongTensor,
                      first_wordpiece_mask: torch.LongTensor,
                      **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        prediction_method = 'argmax' if self.training else self._prediction_method

        mask = self._get_mask(question_and_passage_mask, passage_mask, first_wordpiece_mask)
        masked_indices = mask.nonzero().squeeze()
        masked_log_probs = log_probs[masked_indices]

        if prediction_method == 'viterbi':
            top_two_masked_predicted_tags = torch.Tensor(viterbi_tags(masked_log_probs.unsqueeze(0), 
                                                              transitions=self._transitions, 
                                                              constraint_mask=self._constraint_mask, top_k=2))
            masked_predicted_tags = top_two_masked_predicted_tags[0,0,:]
            if masked_predicted_tags.sum(dim=-1) == 0:
                masked_predicted_tags = top_two_masked_predicted_tags[0,1,:]
        elif prediction_method == 'argmax':
            masked_predicted_tags = torch.argmax(masked_log_probs, dim=-1)
        else:
            raise Exception("Illegal prediction_method")

        masked_qp_indices = np.arange(len(qp_tokens))[masked_indices.cpu()].tolist()

        retrying = False
        while True:
            spans_text, spans_indices = self._decode_spans_from_tags(masked_predicted_tags, masked_qp_indices, 
                                                            qp_tokens, question_passage_wordpieces,
                                                            p_text, q_text)
            if (not retrying and len(spans_text) == 0 and prediction_method=='argmax'):
                retrying = True
                max_start_index = torch.argmax(masked_log_probs[:,self._span_start_label], dim=0)
                masked_predicted_tags[max_start_index] = self._span_start_label
            else:
                break

        if self._unique_decoding:
            spans_text = list(OrderedDict.fromkeys(spans_text))

            if self._substring_unique_decoding:
                spans_text = self._remove_substring_from_decoded_output(spans_text)

        answer_dict = {
            'value': spans_text,
            'spans': spans_indices
        }
        return answer_dict

    def _get_gold_answer(self,
                         gold_answer_representations: Dict[str, torch.LongTensor],
                         log_probs: torch.LongTensor,
                         mask: torch.LongTensor) -> torch.LongTensor:
        answer_as_text_to_disjoint_bios = gold_answer_representations['answer_as_text_to_disjoint_bios']
        answer_as_list_of_bios = gold_answer_representations['answer_as_list_of_bios']
        span_bio_labels = gold_answer_representations['span_bio_labels']

        with torch.no_grad():
            answer_as_list_of_bios = answer_as_list_of_bios * mask.unsqueeze(1)
            if answer_as_text_to_disjoint_bios.sum() > 0:
                # TODO: verify correctness (Elad)

                full_bio = span_bio_labels
                
                if self._generation_top_k > 0:
                    most_likely_predictions, _ = viterbi_decode(log_probs.cpu(), self._bio_allowed_transitions, top_k=self._generation_top_k)
                    most_likely_predictions = torch.FloatTensor(most_likely_predictions).to(log_probs.device)
                    # ^ Should be converted to tensor

                    most_likely_predictions = most_likely_predictions * mask.unsqueeze(1)
                    
                    generated_list_of_bios = self._filter_correct_predictions(most_likely_predictions, answer_as_text_to_disjoint_bios, full_bio)

                    is_pregenerated_answer_format_mask = (answer_as_list_of_bios.sum((1, 2)) > 0).unsqueeze(-1).unsqueeze(-1).long()
                    bio_seqs = torch.cat((answer_as_list_of_bios, (generated_list_of_bios * (1 - is_pregenerated_answer_format_mask))), dim=1)

                    bio_seqs = self._add_full_bio(bio_seqs, full_bio)
                else:
                    is_pregenerated_answer_format_mask = (answer_as_list_of_bios.sum((1, 2)) > 0).long()
                    bio_seqs = torch.cat((answer_as_list_of_bios, (full_bio * (1 - is_pregenerated_answer_format_mask).unsqueeze(-1)).unsqueeze(1)), dim=1)
            else:
                bio_seqs = answer_as_list_of_bios

        return bio_seqs

    def _marginal_likelihood(self,
                             bio_seqs: torch.LongTensor,
                             log_probs: torch.LongTensor):
        # bio_seqs - Shape: (batch_size, # of correct sequences, seq_length)
        # log_probs - Shape: (batch_size, seq_length, 3)

        # Shape: (batch_size, # of correct sequences, seq_length, 3)
        # duplicate log_probs for each gold bios sequence
        expanded_log_probs = log_probs.unsqueeze(1).expand(-1, bio_seqs.size()[1], -1, -1)
        
        # get the log-likelihood per each sequence index
        # Shape: (batch_size, # of correct sequences, seq_length)
        log_likelihoods = \
            torch.gather(expanded_log_probs, dim=-1, index=bio_seqs.unsqueeze(-1)).squeeze(-1)

        # Shape: (batch_size, # of correct sequences)
        correct_sequences_pad_mask = (bio_seqs.sum(-1) > 0).long()

        # Sum the log-likelihoods for each index to get the log-likelihood of the sequence
        # Shape: (batch_size, # of correct sequences)
        sequences_log_likelihoods = log_likelihoods.sum(dim=-1)
        sequences_log_likelihoods = replace_masked_values(sequences_log_likelihoods, correct_sequences_pad_mask, -1e7)

        # Sum the log-likelihoods for each sequence to get the marginalized log-likelihood over the correct answers
        log_marginal_likelihood = logsumexp(sequences_log_likelihoods, dim=-1)

        return log_marginal_likelihood

    def _get_most_likely_likelihood(self,
                             bio_seqs: torch.LongTensor,
                             log_probs: torch.LongTensor):
        # bio_seqs - Shape: (batch_size, # of correct sequences, seq_length)
        # log_probs - Shape: (batch_size, seq_length, 3)

        # Shape: (batch_size, # of correct sequences, seq_length, 3)
        # duplicate log_probs for each gold bios sequence
        expanded_log_probs = log_probs.unsqueeze(1).expand(-1, bio_seqs.size()[1], -1, -1)
        
        # get the log-likelihood per each sequence index
        # Shape: (batch_size, # of correct sequences, seq_length)
        log_likelihoods = \
            torch.gather(expanded_log_probs, dim=-1, index=bio_seqs.unsqueeze(-1)).squeeze(-1)

        # Shape: (batch_size, # of correct sequences)
        correct_sequences_pad_mask = (bio_seqs.sum(-1) > 0).long()

        # Sum the log-likelihoods for each index to get the log-likelihood of the sequence
        # Shape: (batch_size, # of correct sequences)
        sequences_log_likelihoods = log_likelihoods.sum(dim=-1)
        sequences_log_likelihoods = replace_masked_values(sequences_log_likelihoods, correct_sequences_pad_mask, -1e7)

        most_likely_sequence_index = sequences_log_likelihoods.argmax(dim=-1)

        return sequences_log_likelihoods.gather(dim=1, index=most_likely_sequence_index.unsqueeze(-1)).squeeze(dim=-1)

    def _filter_correct_predictions(self, 
                                    predictions: torch.LongTensor, 
                                    answer_as_text_to_disjoint_bios: torch.LongTensor, 
                                    full_bio: torch.LongTensor):
        texts_count = answer_as_text_to_disjoint_bios.size()[1]
        spans_count = answer_as_text_to_disjoint_bios.size()[2]
        predictions_count = predictions.size()[1]

        expanded_predictions = predictions.unsqueeze(2).unsqueeze(2).repeat(1, 1, texts_count, spans_count, 1)
        expanded_answer_as_text_to_disjoint_bios = answer_as_text_to_disjoint_bios.unsqueeze(1)
        expanded_full_bio = full_bio.unsqueeze(1).unsqueeze(-2).unsqueeze(-2)

        disjoint_intersections = (expanded_predictions == expanded_answer_as_text_to_disjoint_bios) & (expanded_answer_as_text_to_disjoint_bios != 0)
        some_intersection = disjoint_intersections.sum(-1) > 0
        only_full_intersections = (((expanded_answer_as_text_to_disjoint_bios != 0) - disjoint_intersections).sum(-1) == 0) & (expanded_answer_as_text_to_disjoint_bios.sum(-1) > 0)
        valid_texts = (((some_intersection ^ only_full_intersections)).sum(-1) == 0) & (only_full_intersections.sum(-1) > 0)
        correct_mask = ((valid_texts == 1).prod(-1) != 0).long()
        correct_mask &= (((expanded_full_bio != expanded_predictions) & (expanded_predictions != 0)).sum((-1, -2, -3)) == 0).long()

        return predictions * correct_mask.unsqueeze(-1)

    def _add_full_bio(self,
                      correct_most_likely_predictions: torch.LongTensor,
                      full_bio: torch.LongTensor):
        predictions_count = correct_most_likely_predictions.size()[1]

        not_added = ((full_bio.unsqueeze(1) == correct_most_likely_predictions).prod(-1).sum(-1) == 0).long()

        return torch.cat((correct_most_likely_predictions, (full_bio * not_added.unsqueeze(-1)).unsqueeze(1)), dim=1)

    def _decode_spans_from_tags(self, masked_tags, masked_qp_indices, 
                            qp_tokens, qp_wordpieces, 
                            passage_text, question_text):
        """
        decoding_style: str - The options are:
                    "single_word_representation" - Each word's wordpieces are aggregated somehow.
                                                    The only decoding_style that requires masking of wordpieces.
                    "at_least_one" - If at least one of the wordpieces is tagged with B,
                                    then the whole word is taken. This is approach yielding the best results
                                    for the non-masked wordpieces models.
                    "forget_wordpieces" - Each wordpiece is regarded as an independent token, 
                                    which means partial words predictions are valid. This is the most natural decoding.
                    "strict_wordpieces" - all of the wordpieces should be tagged as they would have been in the reader
        """
        decoding_style = self._decoding_style
        labels = self._labels
        labels_scheme = self._labels_scheme

        ingested_token_indices = []
        spans_tokens = []
        prev = labels['O']
        current_tokens = []

        context = ''
        for i in range(len(masked_qp_indices)):
            tag = masked_tags[i]
            token_index = masked_qp_indices[i]
            token = qp_tokens[token_index]
            relevant_wordpieces = qp_wordpieces[token_index]

            if token_index in ingested_token_indices:
                continue

            if decoding_style == 'single_word_representation' or decoding_style =='at_least_one':
                tokens = [qp_tokens[j] for j in qp_wordpieces[token_index]]
                token_indices = qp_wordpieces[token_index]
            elif decoding_style == 'forget_wordpieces':
                tokens = [qp_tokens[token_index]]
                token_indices = [token_index]
            elif decoding_style == 'strict_wordpieces':
                num_of_prev_wordpieces = len(relevant_wordpieces) - 1
                if len(relevant_wordpieces) == 1:
                    tokens = [qp_tokens[token_index]]
                elif (token_index == relevant_wordpieces[-1] and # if token is the last wordpiece
                    len(ingested_token_indices) >= len(relevant_wordpieces) - 1 # and the number of ingested is at least the number of previous wordpieces
                    and ingested_token_indices[-num_of_prev_wordpieces:] == relevant_wordpieces[:-1]): # and all the last ingested are exactly the previous wordpieces
                        tokens = [qp_tokens[j] for j in qp_wordpieces[token_index]]
                else:
                    tokens = []
                token_indices = [token_index]
            else:
                raise Exception("Illegal decoding_style")

            add_span = False
            ingest_token = False

            if labels_scheme == 'BIO' or labels_scheme == 'IO':
                if tag == labels['I']:
                    if prev != labels['O'] or labels_scheme == 'IO':
                        ingest_token = True
                        prev = labels['I']
                    else:
                        # Illegal I, treat it as O
                        # Won't occur with Viterbi or constrained beam search, only with argmax
                        prev = labels['O']

                elif labels_scheme != 'IO' and tag == labels['B']:
                    add_span = True
                    ingest_token = True
                    prev = labels['B']

                    if decoding_style == 'strict_wordpieces':
                        if token_index != relevant_wordpieces[0]:
                            ingest_token = False
                            prev = labels['O']

                elif tag == labels['O'] and prev != labels['O']:
                    # Examples: "B O" or "B I O"
                    # consume previously accumulated tokens as a span
                    add_span = True
                    prev = labels['O']
            elif labels_scheme == 'BIOUL':
                if tag == labels['I']:
                    if prev == labels['B'] or prev == labels['I']:
                        ingest_token = True
                        prev = labels['I']
                    else:
                        # Illegal I, treat it as O
                        # Won't occur with Viterbi or constrained beam search, only with argmax
                        prev = labels['O']

                elif tag == labels['B']:
                    if prev == labels['O'] or prev == labels['L'] or prev == labels['U']:
                        ingest_token = True
                        prev = labels['B']
                    else:
                        prev = labels['O']

                    if decoding_style == 'strict_wordpieces':
                        if token_index != relevant_wordpieces[0]:
                            ingest_token = False
                            prev = labels['O']

                elif tag == labels['U']:
                    if prev == labels['O'] or prev == labels['L'] or prev == labels['U']:
                        add_span = True
                        ingest_token = True
                        prev = labels['U']
                    else:
                        prev = labels['O']

                    if decoding_style == 'strict_wordpieces':
                        if token_index != relevant_wordpieces[0]:
                            ingest_token = False
                            prev = labels['O']

                elif tag == labels['L']:
                    if prev == labels['I'] or prev == labels['B']:
                        add_span = True
                        ingest_token = True
                        prev = labels['L']
                    else:
                        prev = labels['O']

                    if decoding_style == 'strict_wordpieces':
                        if token_index != relevant_wordpieces[-1]:
                            ingest_token = False
                            prev = labels['O']

                elif tag == labels['O']:
                    prev = labels['O']

            else:
                raise Exception("Illegal labeling scheme")

            if labels_scheme == 'BIOUL' and ingest_token:
                current_tokens.extend(tokens)
                ingested_token_indices.extend(token_indices)
                context = get_token_context(token)
            if add_span and current_tokens:
                context = get_token_context(current_tokens[0])
                # consume previously accumulated tokens as a span
                spans_tokens.append((context, current_tokens))
                # start accumulating for a new span
                current_tokens = []
            if labels_scheme != 'BIOUL' and ingest_token:
                current_tokens.extend(tokens)
                ingested_token_indices.extend(token_indices)

        if current_tokens:
            # Examples: # "B [EOS]", "B I [EOS]"
            context = get_token_context(current_tokens[0])
            spans_tokens.append((context, current_tokens))

        spans_text, spans_indices = decode_token_spans(spans_tokens, passage_text, question_text)

        return spans_text, spans_indices

    @staticmethod
    def _remove_substring_from_decoded_output(spans):
        new_spans = []
        lspans = [s.lower() for s in spans]

        for span in spans:
            lspan = span.lower()

            # remove duplicates due to casing
            if lspans.count(lspan) > 1:
                lspans.remove(lspan)
                continue
                
            # remove some kinds of substrings
            if not any((lspan + ' ' in s or ' ' + lspan in s or lspan + 's' in s or lspan + 'n' in s or (lspan in s and not s.startswith(lspan) and not s.endswith(lspan))) and lspan != s for s in lspans):
                new_spans.append(span)

        return new_spans
