local config = import "../../abstract/model.jsonnet";
config {
    special_numbers:: [100, 1],
    "dataset_reader"+: {
        "type": "tbmse_drop",
        "answer_field_generators": {
            "tagged_answer": $.answer_field_generators.tagged_answer,
            "arithmetic_answer": $.answer_field_generators.arithmetic_answer,
            "count_answer": $.answer_field_generators.count_answer,
            "passage_span_answer": $.answer_field_generators.passage_span_answer,
            "question_span_answer": $.answer_field_generators.question_span_answer
        },
        "pickle"+: {
            "path": "../pickle/drop",
        }
    },
    "model"+: {
        "dataset_name": "drop",
    },
    "train_data_path": "drop_data/drop_dataset_train.json",
    "validation_data_path": "drop_data/drop_dataset_dev.json",
}