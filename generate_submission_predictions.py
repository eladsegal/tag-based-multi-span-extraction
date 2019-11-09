import sklearn # helps with the "cannot load any more object with static TLS" error
import argparse
import json
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from tqdm import tqdm
from allennlp.common.util import import_submodules

# run with the following command in docker for our main models that needs the override:
# python generate_submission_predictions.py --archive_file [model.tar.gz]  --input_file /drop.json  --output_file /results/predictions.json --cuda-device 0 --include-package src -o '{"model": {"multispan_use_prediction_beam_search": true, "multispan_prediction_beam_size": 5}}'

if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_file", type = str, required = True,
                        help = "URL for a trained model file")
    parser.add_argument("--input_file", type=str, required=True,
                        help='path for drop input files')
    parser.add_argument("--output_file", type=str, required=True,
                        help="path for predictions output file")
    parser.add_argument('--include-package', type=str, action='append',
                            default=[], help='additional packages to include')
    parser.add_argument('-o', '--overrides', type=str, default="",
                            help='a JSON structure used to override the experiment configuration')
    cuda_device = parser.add_mutually_exclusive_group(required=False)
    cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

    args = parser.parse_args()

    # Import any additional modules needed (to register custom classes).
    for package_name in getattr(args, 'include_package', ()):
        import_submodules(package_name)

    # Create predictor
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device, overrides=args.overrides)
    predictor = Predictor.from_archive(archive, "machine-comprehension")
    dataset_reader = predictor._dataset_reader

    predictions = {}

    # Run on input file & collect answers
    instances = dataset_reader.read(args.input_file)
    for instance in tqdm(instances):
        prediction = predictor.predict_instance(instance)
        query_id = prediction["query_id"]
        answer = prediction["answer"]
        predictions[query_id] = answer["value"]


    # Write output file
    with open(args.output_file, "w", encoding = "utf8") as fout:
        json.dump(predictions, fout)
