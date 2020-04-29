local config = import 'quoref_model.jsonnet';
config {
    "dataset_reader"+: {
        "answer_generator_names_per_type": {
            "multiple_span": ["passage_span_answer"],
            "single_span": ["passage_span_answer"]
        },
        "pickle"+: {
            "file_name": "single_heads_" + $.pretrained_model,
        }
    },
    "model"+: {
        "heads": {
            "passage_span": $.heads.passage_span,
            "multi_span": $.heads.multi_span
        }
    }
}
