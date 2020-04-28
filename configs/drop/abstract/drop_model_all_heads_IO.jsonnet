local config = import "drop_model.jsonnet";
config {
    "dataset_reader"+: {
        "answer_generator_names_per_type": {
            "multiple_span": ["tagged_answer"],
            "single_span": ["tagged_answer", "passage_span_answer", "question_span_answer"],
            "number": ["arithmetic_answer", "count_answer", "passage_span_answer", "question_span_answer", "tagged_answer"],
            "date": ["arithmetic_answer", "passage_span_answer", "question_span_answer", "tagged_answer"]
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
            "question_span": $.heads.question_span,
            "multi_span": $.heads.multi_span,
            "arithmetic": $.heads.arithmetic,
            "count": $.heads.count,
        }
    }
}
