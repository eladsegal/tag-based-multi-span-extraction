# Tag-based Multi-Span Extraction in Reading Comprehension

This is the code repository for our paper "Tag-based Multi-Span Extraction in Reading Comprehension" by [Avia Efrat*](mailto:aviaefra@mail.tau.ac.il), [Elad Segal*](mailto:eladsegal@mail.tau.ac.il) and [Mor Shoham*](mailto:morshoham@mail.tau.ac.il) (will be published soon).

This work was done as a final project for the Spring 2019 instances of "Advanced Methods in Natural Language Processing" and "Advanced Methods in Machine Learning" at Tel Aviv University.

*Equal Contribution

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
