from typing import Dict, Tuple, Any

import torch

from allennlp.common import Registrable
from allennlp.data.fields import Field

class AnswerFieldGenerator(Registrable):
    def get_answer_fields(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, Field], bool]:
        raise NotImplementedError

    def get_empty_answer_fields(self, *args: Any, **kwargs: Any) -> Dict[str, Field]:
        raise NotImplementedError
