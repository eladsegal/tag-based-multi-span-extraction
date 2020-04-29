local config = import '../abstract/drop_TASE_BIO.jsonnet';

config {
    "pretrained_model": "roberta-large",
    "bert_dim": 1024,
    "iterator"+: {
        "batch_size": 2
    },
    "trainer"+: {
        "optimizer"+: {
            "lr": 1e-05
        },
        "num_steps_to_accumulate": 6
    },
    "dataset_reader"+: {
        "answer_field_generators"+: {
            "tagged_answer"+: {
                "flexibility_threshold": 1
            }
        },
        "pickle"+: {
            "file_name": "multi_head_flex1_" + $.pretrained_model
        }
    }
}
