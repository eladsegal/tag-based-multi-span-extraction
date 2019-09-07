from pytorch_transformers import BertTokenizer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from typing import Dict, List, Union, Tuple, Any
from overrides import overrides

@Tokenizer.register("bert-drop")
class BertDropTokenizer(Tokenizer):
    def __init__(self, pretrained_model: str):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(token) for token in self.tokenizer.tokenize(text)]