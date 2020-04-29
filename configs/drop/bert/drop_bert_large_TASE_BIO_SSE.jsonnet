local config = import '../abstract/drop_TASE_BIO_SSE.jsonnet';

config {
    "pretrained_model": "bert-large-uncased-whole-word-masking",
    "bert_dim": 1024,
    "iterator"+: {
        "batch_size": 2
    },
    "trainer"+: {
        "optimizer"+: {
            "lr": 1e-05
        },
        "num_steps_to_accumulate": 6
    }
}
