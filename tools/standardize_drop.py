import argparse
import os
import json
from src.data.dataset_readers.utils import standardize_text_advanced
from src.data.dataset_readers.drop.drop_utils import standardize_dataset

"""
Run with
python -m tools.standardize_drop input ouptut
"""
def main(args):
    with open(args.input, encoding = 'utf8') as input_file:
        dataset = json.load(input_file)

    dataset = standardize_dataset(dataset, standardize_text_advanced)

    with open(args.output, mode='w', encoding = 'utf8') as output_file:
        json.dump(dataset, output_file, indent=4)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("input", type=str, help="Input dropified file")
    parse.add_argument("output", type=str, help="Output")
    args = parse.parse_args()

    main(args)
