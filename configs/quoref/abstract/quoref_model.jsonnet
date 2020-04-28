local config = import '../../abstract/model.jsonnet';
config {
    "multi_span_ignore_question": true,
    "dataset_reader"+: {
        "type": "tbmse_quoref",
        "answer_field_generators": {
            "tagged_answer": $.answer_field_generators.tagged_answer,
            "passage_span_answer": $.answer_field_generators.passage_span_answer
        },
        "pickle"+: {
            "path": "../pickle/quoref",
        }
    },
    "model"+: {
        "dataset_name": "quoref",
    },
    "train_data_path": "quoref_data/quoref_dataset_train.json",
    "validation_data_path": "quoref_data/quoref_dataset_dev.json"
}