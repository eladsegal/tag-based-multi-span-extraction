local config = import 'quoref_model.jsonnet';
config {
    "dataset_reader"+: {
        "answer_generator_names_per_type": {
            "multiple_span": ["tagged_answer"],
            "single_span": ["tagged_answer", "passage_span_answer"]
        },
        "pickle"+: {
            "file_name": "all_heads_IO_" + $.pretrained_model,
        }
    },
    "multi_span_labels": {
        'O': 0,
        'I': 1
    },
    "model"+: {
        "heads": {
            "passage_span": $.heads.passage_span,
            "multi_span": $.heads.multi_span
        }
    }
}
