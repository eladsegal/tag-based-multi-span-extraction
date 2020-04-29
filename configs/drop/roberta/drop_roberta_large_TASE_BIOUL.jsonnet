local config = import '../abstract/drop_TASE_BIOUL.jsonnet';

config {
    "pretrained_model": "roberta-large",
    "bert_dim": 1024,
    "iterator"+: {
        "batch_size": 2
    },
    "trainer"+: {
        "optimizer"+: {
            "lr": 5e-06
        },
        "num_steps_to_accumulate": 6
    }
}
