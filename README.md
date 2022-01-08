# A Simple and Effective Model for Answering Multi-span Questions

This repository contains the official implementation of the following paper:  
Elad Segal, Avia Efrat, Mor Shoham, Amir Globerson, Jonathan Berant. ["A Simple and Effective Model for Answering Multi-span Questions"](http://arxiv.org/abs/1909.13375). _In EMNLP, 2020_.

### Citation
```
@inproceedings{Segal2020TASE,
  title={A Simple and Effective Model for Answering Multi-span Questions},
  author={Segal, Elad and Efrat, Avia and Shoham, Mor and Globerson, Amir and Berant, Jonathan},
  booktitle={EMNLP},
  year={2020},
}
```

## [DROP Explorer](https://github.com/eladsegal/DROP-explorer)
Use [DROP Explorer](https://github.com/eladsegal/DROP-explorer) to better understand DROP, Quoref and the models' predictions.

## Usage
The commands listed in this section need to be run from the root directory of the repository.

First, install prerequisites with  
```pip install -r requirements.txt```

### Commands
* Train a model:  
```
allennlp train configs/[config file] -s [training_directory] --include-package src
```

* Output predictions by a model:  
```
allennlp predict model.tar.gz drop_data/drop_dataset_dev.json --predictor machine-comprehension \
--cuda-device 0 --output-file predictions.jsonl --use-dataset-reader --include-package src \
-o "{'validation_dataset_reader.pickle.action': 'None'}"
```

* Evaluate a model (unofficial evaluation code, faster):  
```
allennlp evaluate model.tar.gz drop_data/drop_dataset_dev.json --cuda-device 0 --output-file eval.json \
--include-package src -o "{'validation_dataset_reader.pickle.action': 'None'}"
```

* Evaluate a model (official evaluation code, slower):

  1. 
    ```
    python tools/generate_submission_predictions.py --archive_file model.tar.gz \
    --input_file drop_data/drop_dataset_dev.json --cuda-device 0 --output_file predictions.json \
    --include-package src
  ```
  2. 
    ```
    python -m allennlp.tools.drop_eval --gold_path drop_data/drop_dataset_dev.json \
    --prediction_path predictions.json --output_path metrics.json
    ```
  
### Trained Models
- [RoBERTa TASE_IO + SSE](https://drive.google.com/file/d/1k8MFEmmGeUXlBmghAKN8Xl_a6mbFUHdn/view) - Trained on DROP ([config](https://github.com/eladsegal/tag-based-multi-span-extraction/blob/master/configs/drop/roberta/drop_roberta_large_TASE_IO_SSE.jsonnet))
- [RoBERTa TASE_IO](https://drive.google.com/file/d/1VneI-thp4dfTOcqRPv1-Gzq1jmwsvq_i/view) - Trained on DROP ([config](https://github.com/eladsegal/tag-based-multi-span-extraction/blob/master/configs/drop/roberta/drop_roberta_large_TASE_IO.jsonnet))


