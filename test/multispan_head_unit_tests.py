import os
#print(os.environ.get("PATH")) # Run it when the environment is activated and use the related env values to overwrite the PATH variable 
#os.environ['PATH'] = 'C:\\Users\\Elad\\Anaconda3\\envs\\allennlp;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\mingw-w64\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\usr\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Scripts;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\bin;C:\\Users\\Elad\\Anaconda3\\condabin;'

import unittest
from src.multispan_heads import CRFLossBIO, MultiSpanHead
from src.bert_indexer import BertDropTokenIndexer
from src.bert_tokenizer import BertDropTokenizer
from src.nabertplusplus.nabert_reader import NaBertDropReader
from src.nabertplusplus.nabertplusplus import NumericallyAugmentedBERTPlusPlus
from allennlp.data.token_indexers import PretrainedBertIndexer
from overrides import overrides
import os

from src.bert_tokenizer import BertDropTokenizer

from allennlp.data import Vocabulary


class CRFLossBIOUnitTests(unittest.TestCase):
    pretrained_model = "bert-base-uncased"
    token_indexer = BertDropTokenIndexer(pretrained_model)
    tokenizer = tokenizer = BertDropTokenizer(pretrained_model)
    drop_sample_path = os.path.join(os.path.dirname(__file__), 'resources', 'drop_sample.json')        
    drop_sample2_path = os.path.join(os.path.dirname(__file__), 'resources', 'drop_sample2.json')        
    drop_sample3_path = os.path.join(os.path.dirname(__file__), 'resources', 'drop_sample3.json')        

    def test_bio_spans_decoding(self):
        instances_to_read = 1

        reader = NaBertDropReader(self.tokenizer, {'tokens': self.token_indexer}, max_instances = instances_to_read, extra_numbers=[100, 1], answer_types=['multiple_span'])
        instances = reader.read(self.drop_sample_path)

        vocab = Vocabulary() #self.token_indexer.vocab
        multispan_head = CRFLossBIO(bert_dim=512, dropout_prob=0.1)

        self.assertEqual(instances_to_read, len(instances))
        
        instance = instances[0]

        tags = list(instance.fields['span_bio_labels'])

        span_texts, spans_indices, invalid_tokens = MultiSpanHead.decode_spans_from_tags(tags, instance['metadata']['question_passage_tokens'], instance['metadata']['original_passage'], instance['metadata']['original_question'])

        self.assertFalse(invalid_tokens)

        self.assertEqual(2, len(span_texts))
        self.assertTrue('White' in span_texts)
        self.assertTrue('Asian' in span_texts)
        
        self.assertEqual(2, len(spans_indices))

        passage = instance['metadata']['original_passage']

        self.assertEqual('p', spans_indices[0][0])
        span1_start = spans_indices[0][1]
        span1_end = spans_indices[0][2]

        self.assertEqual('p', spans_indices[1][0])
        span2_start = spans_indices[1][1]
        span2_end = spans_indices[1][2]

        spans_from_passage = [passage[span1_start:span1_end], passage[span2_start:span2_end]]
        self.assertTrue('White' in span_texts)
        self.assertTrue('Asian' in span_texts)

    def test_bio_spans_decoding2(self):
        instances_to_read = 1

        reader = NaBertDropReader(self.tokenizer, {'tokens': self.token_indexer}, max_instances = instances_to_read, extra_numbers=[100, 1], answer_types=['multiple_span'])
        instances = reader.read(self.drop_sample2_path)

        vocab = Vocabulary() #self.token_indexer.vocab
        multispan_head = CRFLossBIO(bert_dim=512, dropout_prob=0.1)

        self.assertEqual(instances_to_read, len(instances))
        
        instance = instances[0]

        tags = list(instance.fields['span_bio_labels'])

        span_texts, spans_indices, invalid_tokens = MultiSpanHead.decode_spans_from_tags(tags, instance['metadata']['question_passage_tokens'], instance['metadata']['original_passage'], instance['metadata']['original_question'])

        self.assertFalse(invalid_tokens)

        self.assertEqual(2, len(span_texts))
        self.assertTrue('Karney' in span_texts)
        self.assertTrue('Kasay' in span_texts)
        
        self.assertEqual(2, len(spans_indices))

        passage = instance['metadata']['original_passage']

        self.assertEqual('p', spans_indices[0][0])
        span1_start = spans_indices[0][1]
        span1_end = spans_indices[0][2]

        self.assertEqual('p', spans_indices[1][0])
        span2_start = spans_indices[1][1]
        span2_end = spans_indices[1][2]

        spans_from_passage = [passage[span1_start:span1_end], passage[span2_start:span2_end]]
        self.assertTrue('Karney' in span_texts)
        self.assertTrue('Kasay' in span_texts)

    def test_bio_spans_decoding3(self):
        instances_to_read = 1

        reader = NaBertDropReader(self.tokenizer, {'tokens': self.token_indexer}, max_instances = instances_to_read, extra_numbers=[100, 1], answer_types=['multiple_span'])
        instances = reader.read(self.drop_sample3_path)

        vocab = Vocabulary() #self.token_indexer.vocab
        multispan_head = CRFLossBIO(bert_dim=512, dropout_prob=0.1)

        self.assertEqual(instances_to_read, len(instances))
        
        instance = instances[0]

        tags = list(instance.fields['span_bio_labels'])

        span_texts, spans_indices, invalid_tokens = MultiSpanHead.decode_spans_from_tags(tags, instance['metadata']['question_passage_tokens'], instance['metadata']['original_passage'], instance['metadata']['original_question'])

        self.assertFalse(invalid_tokens)

        self.assertEqual(2, len(span_texts))
        self.assertTrue('100 Peso note' in span_texts)
        self.assertTrue('500 Pesos note' in span_texts)
        
        self.assertEqual(2, len(spans_indices))

        passage = instance['metadata']['original_passage']

        self.assertEqual('p', spans_indices[0][0])
        span1_start = spans_indices[0][1]
        span1_end = spans_indices[0][2]

        self.assertEqual('p', spans_indices[1][0])
        span2_start = spans_indices[1][1]
        span2_end = spans_indices[1][2]

        spans_from_passage = [passage[span1_start:span1_end], passage[span2_start:span2_end]]
        self.assertTrue('100 Peso note' in span_texts)
        self.assertTrue('500 Pesos note' in span_texts)
        
if __name__ == '__main__':
    unittest.main()