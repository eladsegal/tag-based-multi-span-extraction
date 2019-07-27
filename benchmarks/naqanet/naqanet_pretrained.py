from typing import Iterator, List, Dict

import torch
import numpy as np
from allennlp import pretrained
from allennlp.data.dataset import Batch
from allennlp.nn import util

'''
Instructions:
    1. pip install allennlp
    2. set dev_set_path
    3. set model_path [Optional, Default is to use allennlp pretrained model]

Results:
    allennlp pretrained: {'em': 0.46916946308724833, 'f1': 0.5054509228187923}
    A model with same answering abilities as naqanet (PQNC), trained for 18 epochs: { 'em': 0.4480914429530201, 'f1': 0.4826614932885911 }

Training new models:
    There are configutations to get model with less answering abilities if we need:
        PQ - only passage and question spans
        PQC - passage and question spans + counting

    Train by running:
        allennlp train naqanet_PQ.json -s <some_new_folder_that_will_save_the_results>
'''

# Config
dev_set_path = r"C:\drop\drop_dataset\drop_dataset_dev.json"
model_path = None
cuda_device = -1

# Load pretrained naqanet model
predictor = pretrained.load_archive(model_path, cuda_device) if model_path else pretrained.naqanet_dua_2019()

# Load dev set
drop_data_dev = predictor._dataset_reader.read(dev_set_path)
dev_set_size = len(drop_data_dev)
print ("Loaded dev set. Size: ", dev_set_size)

# Make the model run on cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
predictor._model.to(device)
print("Executing the model on :", predictor._model._get_prediction_device())

# Apply on dev set
batch_size = 16

for i in np.arange(0, dev_set_size, batch_size):
    res = predictor.predict_batch_instance(drop_data_dev[i:i+batch_size])

    if (i / batch_size) % 10 == 0:
        print (f'Completed {i} out of {dev_set_size}')

# The naqanet instance calculates exact match and f1 automatically by default
print (predictor._model.get_metrics())

