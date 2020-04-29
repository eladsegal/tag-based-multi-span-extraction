local config = import "drop_model.jsonnet";
config {
    "dataset_reader"+: {
        "answer_generator_names_per_type": {
            "multiple_span": ["passage_span_answer"],
            "single_span": ["passage_span_answer", "question_span_answer"],
            "number": ["arithmetic_answer", "count_answer", "passage_span_answer", "question_span_answer"],
            "date": ["arithmetic_answer", "passage_span_answer", "question_span_answer"]
        },
        "pickle"+: {
            "file_name": "single_heads_" + $.pretrained_model
        }
    },
    "model"+: {
        "heads": {
            "passage_span": $.heads.passage_span,
            "question_span": $.heads.question_span,
            "arithmetic": $.heads.arithmetic,
            "count": $.heads.count
        }
    }
}
