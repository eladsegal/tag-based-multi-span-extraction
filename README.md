# A Simple and Effective Model for Answering Multi-span Questions

This repository contains the official implementation of the following paper:  
Elad Segal*, Avia Efrat*, Mor Shoham*, Amir Globerson, Jonathan Berant. ["A Simple and Effective Model for Answering Multi-span Questions"](http://arxiv.org/abs/1909.13375).

*Equal Contribution

## [DROP Explorer](https://github.com/eladsegal/DROP-explorer)
Use [DROP Explorer](https://github.com/eladsegal/DROP-explorer) to better understand DROP, Quoref and the models' predictions.

## Usage
The commands listed in this section need to be run from the root directory of the repository.

First, install prerequisites with  
```pip install -r requirements.txt```

### Commands
* Train a model:  
```allennlp train configs/[config file] -s [training_directory] --include-package src```

* Output predictions by a model:  
```allennlp predict model.tar.gz drop_data/drop_dataset_dev.json --predictor machine-comprehension --cuda-device 0 --output-file predictions.jsonl --use-dataset-reader --include-package src```

* Evaluate a model (unofficial evaluation code, fast):  
```allennlp evaluate model.tar.gz drop_data/drop_dataset_dev.json --cuda-device 0 --output-file eval.json --include-package src```

* Evaluate a model (official evaluation code, slow):
  1. ```python tools/generate_submission_predictions.py --archive_file model.tar.gz --input_file drop_data/drop_dataset_dev.json --cuda-device 0 --output_file predictions.json --include-package src```
  2. ```python -m allennlp.tools.drop_eval --gold_path drop_data/drop_dataset_dev.json --prediction_path predictions.json --output_path metrics.json```
