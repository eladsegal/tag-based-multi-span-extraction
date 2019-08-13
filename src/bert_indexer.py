from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from pytorch_pretrained_bert import BertTokenizer
from typing import Dict, List, Union, Tuple, Any
from overrides import overrides

@TokenIndexer.register("multi_span_drop")
class BertDropTokenIndexer(WordpieceIndexer):
    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512) -> None:
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        super().__init__(vocab=bert_tokenizer.vocab,
                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                         max_pieces=max_pieces,
                         namespace="bert",
                         separator_token="[SEP]")    