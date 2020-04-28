local config = import 'quoref_model.jsonnet';
config {
    "dataset_reader"+: {
        "answer_generator_names_per_type": {
            "multiple_span": ["tagged_answer"],
            "single_span": ["tagged_answer"]
        },
        "pickle"+: {
            "file_name": "multi_head_" + $.pretrained_model,
        }
    },
    "model"+: {
        "heads": {
            "multi_span": $.heads.multi_span
        }
    }
}
