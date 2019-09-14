#!/bin/sh
set -e
ARCHIVE_FILE="model4_large_wmfs.tar.gz"
python generate_submission_predictions.py --archive_file $ARCHIVE_FILE --input_file /drop.json  --output_file /results/predictions.json --include-package src -o '{"model": {"multispan_use_prediction_beam_search": true, "multispan_prediction_beam_size": 5}}'