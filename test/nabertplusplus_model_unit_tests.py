import unittest
from src.bert_indexer import BertDropTokenIndexer
from src.bert_tokenizer import BertDropTokenizer
from src.nabertplusplus.nabert_reader import NaBertDropReader
from src.nabertplusplus.nabertplusplus import NumericallyAugmentedBERTPlusPlus
from allennlp.data.token_indexers import PretrainedBertIndexer
from overrides import overrides
import os
from pytorch_pretrained_bert import BertTokenizer

from allennlp.data import Vocabulary

class NabertPlusPlusModelUnitTests(unittest.TestCase):
    pretrained_model = "bert-base-uncased"    
    token_indexer = BertDropTokenIndexer(pretrained_model)
    tokenizer = tokenizer = BertDropTokenizer(pretrained_model)
    
    drop_sample_path = os.path.join(r"data\drop_dataset_train.json")        

    def test_nabertplusplus_sanity(self):
        instances_to_read = 12

        reader = NaBertDropReader(self.tokenizer, {'tokens': self.token_indexer}, max_instances = instances_to_read, extra_numbers=[100, 1])
        instances = reader._read(self.drop_sample_path)

        vocab = Vocabulary() 
        model = NumericallyAugmentedBERTPlusPlus(vocab, self.pretrained_model, special_numbers=[100, 1])
        model.eval()
        outputs = model.forward_on_instances(instances)

        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()