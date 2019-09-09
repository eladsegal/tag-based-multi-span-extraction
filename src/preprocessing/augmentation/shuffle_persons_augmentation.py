from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

from src.multispan_heads import decode_token_spans

from src.preprocessing.utils import get_all_subsequences

import random
import torch

class ShufflePersonAugmentation:
    '''
    Performs augmentation for multi span questions where all answers are tagged as PER by shuffling the persons in the passage.
    
    Parameters
    ----------
    augmentation_ratio : ``int`` optional (default = 1)
        If 1, each valid question will be augmented once.
        If less than once, then the ratio is the probability of the valid question to be augmented. 
    tagger : ``Predictor``, optional (default = named_entity_recognition_with_elmo_peters_2018)
        The maximum number of decoding steps to take, i.e. the maximum length
    '''    

    def __init__(self, augmentation_ratio = 1, tagger = None): 
        self._augmentation_ratio = augmentation_ratio
        
        if tagger is not None:
            self._tagger = tagger
        else:
            cuda_device = 0 if torch.cuda.is_available() else -1            
            archive = load_archive('https://allennlp.s3.amazonaws.com/models/ner-model-2018.12.18.tar.gz', cuda_device = cuda_device)
            self._tagger = Predictor.from_archive(archive, 'sentence-tagger')
            self._tagger._dataset_reader._token_indexers['token_characters']._min_padding_length = 3

        self._cached_passage_id = ""
        self._cached_passage_tags = None
        self._reconstructed_passage_text = ""

    def augment(self, passage_id, question_id, passage_text, answer_texts):
        
        # Valid for augmentation
        if self._augmentation_ratio < 1 and random.uniform(0, 1) > self._augmentation_ratio:
            return []
        
        # Only multi span questions are augmented
        if answer_texts is None or len(answer_texts) <= 1:
            return []
        
        if passage_id == self._cached_passage_id:
            passage_tags = self._cached_passage_tags
            reconstructed_passage_text = self._reconstructed_passage_text
        else:
            passage_tags = self._tagger.predict_json({"sentence": passage_text})
            passage_tags['idxs'] = []
            
            temp_passage = passage_text
            reconstructed_passage_text = ''

            absolute_index = 0
            for i, word in enumerate(passage_tags['words']):
                tag = passage_tags['tags'][i]

                relative_index = temp_passage.index(word)
                first_part = temp_passage[:relative_index]
                second_part = word
                reconstructed_passage_text += first_part + second_part

                start_idx = absolute_index + relative_index
                end_idx = start_idx + len(word) # exclusive
                passage_tags['idxs'].append((start_idx, end_idx)) 

                absolute_index = end_idx
                temp_passage = temp_passage[relative_index + len(word):]
            
            self._cached_passage_id = passage_id
            self._cached_passage_tags = passage_tags
            self._reconstructed_passage_text = reconstructed_passage_text

        if reconstructed_passage_text != passage_text:
            return []

        # Don't try augmentation if there are shared words between the answers or repeating words in an answer
        words_set = set()
        words_count = 0
        for answer_text in answer_texts:
            words = answer_text.split()
            words_set.update(words)
            words_count += len(words)

            if len(words_set) != words_count:
                return []
        

        # Validations
        spans = DropReader.find_valid_spans([Token(word) for word in passage_tags['words']] , answer_texts)
       
        # Validate each answer has a span (tokenizing here is by the tagger so we can't assume our fixes to the data are enough)    
        for answer_text in answer_texts:
            answer_has_tag = False

            for span in spans:            
                span_text = ' '.join(passage_tags['words'][span[0]:span[1]+1]).lower()            

                if answer_text.lower() == span_text:
                    answer_has_tag = True
                    break
            
            if not answer_has_tag:
                return []

        # Validate all spans have PER tags and that there is no span with PER immediately before or after to avoid replacement of partial names
        for span in spans:            
            answer_tags = passage_tags['tags'][span[0]:span[1]+1]
            span_length = len(answer_tags)

            # if not all(tag.endswith('PER') for tag in answer_tags): # # Should we enforce a proper BILOU tagging sequence?
            if not self.is_valid_BILOU(answer_tags):
                return []
            
            # Check if we need it
            '''if span[0] > 0 and passage_tags['tags'][span[0] - 1].endswith('PER'):
                return []

            if span[1] < len(passage_tags) - 1 and passage_tags['tags'][span[1] + 1].endswith('PER'):
                return []'''


        # Heavier stuff, after fast validations
        new_passage_text = passage_text

        PLACEHOLDER_SYMBOL = "#" 
        pending_swaps = []
        swaps_mapping = [(i + 1) % len(answer_texts) for i in range(len(answer_texts))]

        subs_per_answer_text = [get_all_subsequences(answer_text.split()) for answer_text in answer_texts]
        for i, answer_text in enumerate(answer_texts):
            replacer_answer_index = swaps_mapping[i]
            for sub in subs_per_answer_text[i]:
                spans_per_sub = DropReader.find_valid_spans([Token(word) for word in passage_tags['words']] , [sub])
                relative_start_idx = answer_text.index(sub)
                relative_end_idx = relative_start_idx + len(sub) # exclusive
                if (len(spans_per_sub) > 0) and answer_text.index(sub) != 0 and (relative_end_idx != len(answer_text)):
                    # We have a span that is from the middle of the answer. Too ambiguous, ignore question
                    return []
                partition = 1
                if relative_start_idx == 0 and relative_end_idx == len(answer_text):
                    partition = -1
                elif relative_start_idx == 0:
                    partition = 0

                for span in spans_per_sub:
                    answer_tags = passage_tags['tags'][span[0]:span[1]+1]
                    # if not all(tag.endswith('PER') for tag in answer_tags): # Should we enforce a proper BILOU tagging sequence?
                    if not self.is_valid_BILOU(answer_tags):
                        continue
                        
                    first_pard_end = passage_tags['idxs'][span[0]][0]
                    last_part_start = passage_tags['idxs'][span[1]][1]
                    pending_swaps.append({'replacer_answer_index': replacer_answer_index, 
                                        'partition':partition,
                                        'first_pard_end': first_pard_end,
                                        'last_part_start': last_part_start
                                        })
                    first_part = new_passage_text[:passage_tags['idxs'][span[0]][0]]
                    last_part = new_passage_text[passage_tags['idxs'][span[1]][1]:]
                    new_passage_text = first_part + (PLACEHOLDER_SYMBOL) * (last_part_start - first_pard_end)  + last_part
                    
        for swap in sorted(pending_swaps, key=lambda x: x['first_pard_end'], reverse=True):
            replacer_answer_index = swap['replacer_answer_index']
            partition = swap['partition']
            first_pard_end = swap['first_pard_end']
            last_part_start = swap['last_part_start']

            replacer_answer = answer_texts[replacer_answer_index]

            if partition == -1:
                replacer = replacer_answer
            else:
                answer_parts = replacer_answer.split()
                if len(answer_parts) > 1:
                    replacer = answer_parts[0] if partition == 0 else ' '.join(answer_parts[1:])
                else:
                    replacer = replacer_answer

            first_part = new_passage_text[:first_pard_end]
            last_part = new_passage_text[last_part_start:]

            new_passage_text = first_part + replacer + last_part

        return [new_passage_text]

    @staticmethod
    def is_valid_BILOU(answer_tags):
        span_length = len(answer_tags)

        is_start_valid = (answer_tags[0] == 'B-PER' and span_length > 1) or (answer_tags[0] == 'U-PER' and span_length == 1)
        is_middle_valid = all(tag == 'I-PER' for tag in answer_tags[1:-1])
        is_end_valid = (answer_tags[span_length - 1] == 'L-PER' and span_length > 1) or (answer_tags[0] == 'U-PER' and span_length == 1)

        return is_start_valid and is_middle_valid and is_end_valid 
