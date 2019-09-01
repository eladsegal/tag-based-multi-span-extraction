DEBUG = True
if DEBUG:
    import ptvsd
    ptvsd.enable_attach(address=('localhost', 5679), redirect_output=True)
    ptvsd.wait_for_attach()

import json
import shutil
import sys

from allennlp.commands import main

config_file = "configs/all_heads_mos.json"

overrides_dict = {}
USE_CPU = False # Use overrides to train on CPU.
overrides_dict['train_data_path'] = 'data/drop_dataset_train_sample.json'
overrides_dict['validation_data_path'] = 'data/drop_dataset_dev_sample.json'



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

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "src",
    "-o", overrides,
]

main()