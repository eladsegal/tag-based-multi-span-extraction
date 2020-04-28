local config = import "drop_model.jsonnet";
config {
    "dataset_reader"+: {
        "answer_generator_names_per_type": {
            "multiple_span": ["tagged_answer"],
            "single_span": ["tagged_answer"],
            "number": ["arithmetic_answer", "count_answer", "tagged_answer"],
            "date": ["arithmetic_answer", "tagged_answer"]
        },
        "pickle"+: {
            "file_name": "multi_head_BIOUL_" + $.pretrained_model
        }
    },
    "multi_span_labels": {
        'O': 0,
        'B': 1,
        'I': 2,
        'U': 3,
        'L': 4
    },
    "model"+: {
        "heads": {
            "multi_span": $.heads.multi_span,
            "arithmetic": $.heads.arithmetic,
            "count": $.heads.count
        }
    }
}
