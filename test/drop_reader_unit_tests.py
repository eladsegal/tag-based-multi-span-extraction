import unittest
from src.bert_tokenizer import BertDropTokenizer
from src.bert_reader import BertDropReader
from allennlp.data.token_indexers import PretrainedBertIndexer
from overrides import overrides
import os
from pytorch_pretrained_bert import BertTokenizer

class DropReaderUnitTests(unittest.TestCase):
    pretrained_model = "bert-base-cased"
    token_indexer = PretrainedBertIndexer(pretrained_model=pretrained_model, max_pieces=512, do_lowercase=False)
    tokenizer = tokenizer = BertDropTokenizer(pretrained_model)
    drop_sample_path = os.path.join(os.path.dirname(__file__), 'resources', 'drop_sample.json')    

    def test_read_instances_from_correct_type(self):
        instances_to_read = 1

        reader = BertDropReader(self.tokenizer, {'bert_indexer': self.token_indexer}, max_instances = instances_to_read, answer_types=['multiple_span'])
        instances = reader._read(self.drop_sample_path)

        self.assertEqual(instances_to_read, len(instances))
        self.assertEqual('multiple_span', instances[0]['metadata']['answer_type'])

        print(instances[0])
        

if __name__ == '__main__':
    unittest.main()





