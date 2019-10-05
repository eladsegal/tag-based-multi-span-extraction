# Tag-based Multi-Span Extraction in Reading Comprehension

This is the official code repository for "Tag-based Multi-Span Extraction in Reading Comprehension" ([preprint](http://arxiv.org/abs/1909.13375)) by [Avia Efrat](mailto:aviaefra@mail.tau.ac.il)\*, [Elad Segal](mailto:eladsegal@mail.tau.ac.il)\* and [Mor Shoham](mailto:morshoham@mail.tau.ac.il)\*.  
NABERT+ ([raylin1000/drop-bert](https://github.com/raylin1000/drop-bert/)) by Kinley and Lin was used as a basis for our work.

This work was done as a final project for the spring 2019 instances of "Advanced Methods in Natural Language Processing" and "Advanced Methods in Machine Learning" at Tel Aviv University.

*Equal Contribution

## [DROP Explorer](https://github.com/eladsegal/DROP-explorer)
Use [DROP Explorer](https://github.com/eladsegal/DROP-explorer) to better familiarize yourself with DROP and the models' predictions.

## Usage
The commands listed in this section need to be run from the root directory of the repository.

First, install prerequisites with  
```pip install -r requirements.txt```

### Commands
* Train a model:  
```allennlp train configs/[config file] -s [training_directory] --include-package src```

* Output predictions by a model:  
```allennlp predict model.tar.gz data/drop_dataset_dev.json --predictor machine-comprehension --cuda-device 0 --output-file predictions.jsonl --use-dataset-reader --include-package src```

* Evaluate a model (unofficial evaluation code, fast):  
```allennlp evaluate model.tar.gz data/drop_dataset_dev.json --cuda-device 0 --output-file eval.json --include-package src```

* Evaluate a model (official evaluation code, slow):
  1. ```python generate_submission_predictions.py --archive_file model.tar.gz --input_file data/drop_dataset_dev.json --cuda-device 0 --output_file predictions.json --include-package src```
  2. ```python -m allennlp.tools.drop_eval --gold_path data/drop_dataset_dev.json --prediction_path predictions.json --output_path metrics.json```
