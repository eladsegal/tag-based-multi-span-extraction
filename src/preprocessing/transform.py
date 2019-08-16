import json

from src.preprocessing.annotate import ERROR_TYPES, ERRORS_KEYNAME
from src.preprocessing.utils import save_dataset, load_dataset, deep_dict_update


def create_dataset_with_no_errors(original_dataset_path, output_path=None):
    if output_path is None:
        output_path = original_dataset_path

    dataset = load_dataset(original_dataset_path)

    passages_where_all_the_questions_have_errors = set()

    for passage_id, passage_data in dataset.items():
        passage_data["qa_pairs"] = [qa for qa in passage_data["qa_pairs"] if not _get_errors(qa)]

        if len(passage_data["qa_pairs"]) == 0:
            passages_where_all_the_questions_have_errors.add(passage_id)

    dataset = {k: v for k, v in dataset.items() if k not in passages_where_all_the_questions_have_errors}

    save_dataset(dataset, output_path)


def create_dataset_with_error_type(original_dataset_path, error_type, output_path=None):

    if error_type not in ERROR_TYPES:
        raise Exception(f'Invalid error type. Valid types: {ERROR_TYPES}')

    if output_path is None:
        output_path = original_dataset_path.replace('.json', f'_with_{error_type}_errors.json')

    dataset = load_dataset(original_dataset_path)

    passages_no_question_of_the_error_type = set()

    for passage_id, passage_data in dataset.items():
        qas_with_error_type = [qa for qa in passage_data["qa_pairs"] if error_type in _get_errors(qa)]
        passage_data["qa_pairs"] = qas_with_error_type

        if len(passage_data["qa_pairs"]) == 0:
            passages_no_question_of_the_error_type.add(passage_id)

    dataset = {k: v for k, v in dataset.items() if k not in passages_no_question_of_the_error_type}

    save_dataset(dataset, output_path)


def apply_fixes(dataset, fixes, output_path):
    dataset = load_dataset(dataset)
    with open(fixes) as dataset_file:
        fixes = json.load(dataset_file)

    for passage_id, fixes in fixes.items():
        org_qa = dataset[passage_id]['qa_pairs']

        # updates
        for str_idx, update in fixes.get('updates', {}).items():
            deep_dict_update(org_qa[int(str_idx)], update)

        # deletions
        indices_to_delete = fixes.get('deletions', [])
        for index in reversed(indices_to_delete):
            del org_qa[index]

    save_dataset(dataset, output_path)


def _get_errors(qa):
    return qa['answer'].get(ERRORS_KEYNAME, {})
