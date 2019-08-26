import os
#print(os.environ.get("PATH")) # Run it when the environment is activated and use the related env values to overwrite the PATH variable 
os.environ['PATH'] = 'C:\\Users\\Elad\\Anaconda3\\envs\\allennlp;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\mingw-w64\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\usr\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Scripts;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\bin;C:\\Users\\Elad\\Anaconda3\\condabin;'

import unittest
from src.bert_tokenizer import BertDropTokenizer
from src.bert_reader import BertDropReader
from allennlp.data.token_indexers import PretrainedBertIndexer
from overrides import overrides

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





