DEBUG = True
if DEBUG:
    import ptvsd
    ptvsd.enable_attach(address=('localhost', 5679), redirect_output=True)
    ptvsd.wait_for_attach()

import json
import shutil
import sys

from allennlp.commands import main

config_file = "configs/all_heads_mos_flexible_loss.json"

overrides_dict = {}
overrides_dict['train_data_path'] = 'data/drop_dataset_train_sample.json'
overrides_dict['validation_data_path'] = 'data/drop_dataset_dev_sample.json'
#overrides_dict['dataset_reader'] = {"max_instances": 2}


USE_CPU = False # Use overrides to train on CPU.
if USE_CPU:
    overrides_dict['trainer'] = {"cuda_device": -1}

overrides = json.dumps(overrides_dict)


serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Commands examples:
# allennlp train configs/[config file] -s temp --include-package src
# allennlp evaluate model.tar.gz data/drop_dataset_dev.json --cuda-device 0 --output-file eval.json --include-package src

# Assemble the command into sys.argv
'''sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "src",
    "-o", overrides
]'''

sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    "../model_flexible_loss_13.tar.gz",
    "data/drop_dataset_dev.json",
    "--cuda-device", "0",
    "--include-package", "src",
    "--output-file", "eval.json"
]

main()