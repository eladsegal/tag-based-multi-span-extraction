# Tag-based Multi-Span Extraction in Reading Comprehension

This is the code repository for our paper "Tag-based Multi-Span Extraction in Reading Comprehension" (will be published soon).

## Usage
The commands listed in this section needs to be run from the root directory of the repository.

First, install prerequisites with 

```pip install -r requirements.txt```

### Commands
* Train a model:

```allennlp train configs/[config file] -s [training_directory] --include-package src```

* Output predictions by a model: 

```allennlp predict model.tar.gz data/drop_dataset_dev.json --predictor machine-comprehension --cuda-device 0 --output-file predictions.jsonl --use-dataset-reader --include-package src```

* Evaluate a model:

```allennlp evaluate model.tar.gz data/drop_dataset_dev.json --cuda-device 0 --output-file eval.json --include-package src```
