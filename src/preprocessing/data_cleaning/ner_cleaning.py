from allennlp import pretrained
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from overrides import overrides
import os
import json
import torch

from src.preprocessing.utils import get_answer_type
from src.preprocessing.data_cleaning.remove_non_number_spans import RemoveNonNumberSpans
from src.preprocessing.data_cleaning.remove_non_per_spans import RemoveNonPerSpans
from src.preprocessing.data_cleaning.remove_non_org_spans import RemoveNonOrgLocMiscSpans
from src.preprocessing.data_cleaning.trim_spans import TrimSpans

from tqdm import tqdm

'''
General instructions:
Write a class that implement CleaningObjective and add it with\instead the current obejctives.
Run once and observe the expected changes in cleaning_info.json.
If it looks good then drop_dataset_train_cleaned is clean.
If there are only few example that doesn't look good, whitelist them in the cleaning objective and rerun.
'''

if __name__ == "__main__":
    debug_question = None

    # Define dataset cleaning objectives
    cleaning_objectives = [
        RemoveNonPerSpans(),
        RemoveNonOrgLocMiscSpans(),
        RemoveNonNumberSpans()
    ]

    # Define output path
    out_dir = os.path.join(os.path.dirname(__file__), 'cleaning_logs')
    cleaning_info_path = 'cleaning_info.json'
    cleaning_dataset_path = 'drop_dataset_train_clean.json'

    # Load dataset
    drop_path = os.path.join(r"data", "drop_dataset_train.json")

    with open(drop_path, 'r') as f:
        drop = json.load(f)

    # Load NER tagger
    cuda_device = 0 if torch.cuda.is_available() else -1
    print('Running on device: ', cuda_device)
    archive = load_archive('https://allennlp.s3.amazonaws.com/models/ner-model-2018.12.18.tar.gz', cuda_device = cuda_device)
    predictor = Predictor.from_archive(archive, 'sentence-tagger')
    predictor._dataset_reader._token_indexers['token_characters']._min_padding_length = 3

    # Clean the dataset   
    cleaning_info = dict()

    for passage_id, data in tqdm(drop.items()):
        ner_tagging_passage = None

        for qa_pair in data['qa_pairs']:
            if debug_question is not None and debug_question != qa_pair['query_id']:
                continue

            for co in cleaning_objectives:
                if qa_pair['query_id'] in co.whitelist:
                    continue

                ner_tagging_question = None

                if not co.is_fitting_objective(data['passage'], qa_pair['question'], qa_pair['answer']):
                    continue

                if ner_tagging_passage is None:
                    ner_tagging_passage = predictor.predict_json({"sentence": data['passage']})

                if ner_tagging_question is None:
                    ner_tagging_question = predictor.predict_json({"sentence": qa_pair['question']})            

                clean_result = co.clean(data['passage'], qa_pair['question'], qa_pair['answer'], ner_tagging_passage, ner_tagging_question)

                if clean_result is None:
                    continue

                if passage_id not in cleaning_info:
                    cleaning_info[passage_id] = dict()

                if qa_pair['query_id'] not in cleaning_info[passage_id]:
                    cleaning_info[passage_id][qa_pair['query_id']] = dict()

                cleaning_info[passage_id][qa_pair['query_id']]['question'] = qa_pair['question']
                cleaning_info[passage_id][qa_pair['query_id']][co.name] = dict()

                if 'passage' in clean_result:
                    # log the change
                    cleaning_info[passage_id][qa_pair['query_id']][co.name]['old_passage'] = data['passage']
                    cleaning_info[passage_id][qa_pair['query_id']][co.name]['new_passage'] = clean_result['passage']

                    # perform the clean
                    data['passage'] = clean_result['passage']

                if 'question' in clean_result:
                    # log the change
                    cleaning_info[passage_id][qa_pair['query_id']][co.name]['old_question'] = qa_pair['question']
                    cleaning_info[passage_id][qa_pair['query_id']][co.name]['new_question'] = clean_result['question']

                    # perform the clean
                    qa_pair['question'] = clean_result['question']

                if 'answer' in clean_result:
                    # log the change
                    cleaning_info[passage_id][qa_pair['query_id']][co.name]['old_answer'] = qa_pair['answer']
                    cleaning_info[passage_id][qa_pair['query_id']][co.name]['new_answer'] = clean_result['answer']

                    # perform the clean
                    qa_pair['answer'] = clean_result['answer']

    with open(os.path.join(out_dir, cleaning_info_path), 'w') as f:
        json.dump(cleaning_info, f, indent=4)

    with open(os.path.join(out_dir, cleaning_dataset_path), 'w') as f:
        json.dump(drop, f, indent=4)
            