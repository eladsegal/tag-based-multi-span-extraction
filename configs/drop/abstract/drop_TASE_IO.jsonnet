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
            "file_name": "multi_head_IO_" + $.pretrained_model
        }
    },
    "multi_span_labels": {
        'O': 0,
        'I': 1
    },
    "model"+: {
        "heads": {
            "multi_span": $.heads.multi_span,
            "arithmetic": $.heads.arithmetic,
            "count": $.heads.count
        }
    }
}
