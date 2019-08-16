from src.preprocessing.utils import save_dataset, load_dataset


def filter_all_errors(dataset_path, output_path=None):
    if output_path is None:
        output_path = dataset_path

    dataset = load_dataset(dataset_path)

    passages_where_all_the_questions_have_errors = set()

    for passage_id, passage_data in dataset.items():
        passage_data["qa_pairs"] = [qa for qa in passage_data["qa_pairs"] if not qa.get('errors')]

        if len(passage_data["qa_pairs"]) == 0:
            passages_where_all_the_questions_have_errors.add(passage_id)

    dataset = {k: v for k, v in dataset.items() if k not in passages_where_all_the_questions_have_errors}

    save_dataset(dataset, output_path)
