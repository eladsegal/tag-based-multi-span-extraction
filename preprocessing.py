import os

from drop_bert.data_processing import BertDropTokenIndexer
from src.bert_reader import BertDropReader
from src.bert_tokenizer import BertDropTokenizer

tokenizer = BertDropTokenizer('bert-base-cased')
indexer = BertDropTokenIndexer('bert-base-cased')
token_indexers = {'main_indexer': indexer}
reader = BertDropReader(tokenizer, token_indexers)

TRAIN_PATH = os.path.join('drop_dataset', 'drop_dataset_train.json')
result = reader.read(TRAIN_PATH)