"""
A `Flask <http://flask.pocoo.org/>`_ server for serving predictions
from a single AllenNLP model. It also includes a very, very bare-bones
web front-end for exploring predictions (or you can provide your own).

For example, if you have your own predictor and model in the `my_stuff` package,
and you want to use the default HTML, you could run this like

```
python -m allennlp.service.server_simple \
    --archive-path ~/Desktop/drop_bert/model.tar.gz \
    --predictor machine-comprehension \
    --additional-path ~/Desktop/drop_bert
    --include-package drop_bert
```
"""
from typing import List, Callable
import argparse
import json
import logging
import os
from string import Template
import sys

from flask import Flask, request, Response, jsonify, send_file, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from allennlp.common import JsonDict
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['message'] = self.message
        return error_dict


def make_app(predictor: Predictor,
             data_path: str,
             static_dir: str = None,
             sanitizer: Callable[[JsonDict], JsonDict] = None) -> Flask:
    """
    Creates a Flask app that serves up the provided ``Predictor``
    along with a front-end for interacting with it.

    If you want to use the built-in bare-bones HTML, you must provide the
    field names for the inputs (which will be used both as labels
    and as the keys in the JSON that gets sent to the predictor).

    If you would rather create your own HTML, call it index.html
    and provide its directory as ``static_dir``. In that case you
    don't need to supply the field names -- that information should
    be implicit in your demo site. (Probably the easiest thing to do
    is just start with the bare-bones HTML and modify it.)

    In addition, if you want somehow transform the JSON prediction
    (e.g. by removing probabilities or logits)
    you can do that by passing in a ``sanitizer`` function.
    """
    if static_dir is not None:
        static_dir = os.path.abspath(static_dir)
        if not os.path.exists(static_dir):
            logger.error("app directory %s does not exist, aborting", static_dir)
            sys.exit(-1)
    elif static_dir is None:
        print("build_dir was not passed. Demo won't render on this port.\n"
              "You must use nodejs + react app to interact with the server.")

    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response: # pylint: disable=unused-variable
        if static_dir is not None:
            return send_file(os.path.join(static_dir, 'index.html'))
        else:
            return Response(response=f'In order to use this server, please run "npm start" from visualizer/frontend', status=200)

    def get_dataset_list():
        return [f[:-len('.json')] for f in os.listdir(data_path) if f.endswith('.json')]

    @app.route('/dataset-list', methods=['GET', 'OPTIONS'])
    def dataset_list() -> Response:  # pylint: disable=unused-variable
        """return the named dataset"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        json_files = get_dataset_list()
        return jsonify(json_files)

    @app.route('/dataset', methods=['GET', 'OPTIONS'])
    def dataset() -> Response:  # pylint: disable=unused-variable
        """return the named dataset"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        name = request.args.get('name')

        if name not in get_dataset_list():
            return jsonify(error=f'Dataset {name} does not exist.')
        return send_from_directory(data_path, f'{name}.json')

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        prediction = predictor.predict_json(data)
        if sanitizer is not None:
            prediction = sanitizer(prediction)

        log_blob = {"inputs": data, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))

        return jsonify(prediction)

    @app.route('/predict_batch', methods=['POST', 'OPTIONS'])
    def predict_batch() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        prediction = predictor.predict_batch_json(data)
        if sanitizer is not None:
            prediction = [sanitizer(p) for p in prediction]

        return jsonify(prediction)

    @app.route('/<path:path>')
    def static_proxy(path: str) -> Response: # pylint: disable=unused-variable
        if static_dir is not None:
            return send_from_directory(static_dir, path)
        else:
            raise ServerError("static_dir not specified", 404)

    return app

def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_path,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    return Predictor.from_archive(archive, args.predictor)

def main(args):
    # Executing this file with no extra options runs the simple service with the bidaf test fixture
    # and the machine-comprehension predictor. There's no good reason you'd want
    # to do this, except possibly to test changes to the stock HTML).

    parser = argparse.ArgumentParser(description='Serve up a simple model')

    parser.add_argument('--archive-path', type=str, help='path to trained archive file')
    parser.add_argument('--predictor', type=str, help='name of predictor')
    parser.add_argument('--weights-file', type=str,
                        help='a path that overrides which weights file to use')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--static-dir', type=str, help='serve index.html from this directory')
    parser.add_argument('--data-path', type=str, help='get data from this path')
    parser.add_argument('--port', type=int, default=8000, help='port to serve the demo on')

    parser.add_argument('--additional-path',
                        type=str,
                        action='append',
                        default=[],
                        help='additional path to add')
    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')

    args = parser.parse_args(args)

    # Additional paths
    for path in args.additional_path:
        sys.path.append(os.path.abspath(path))

    # Load modules
    for package_name in args.include_package:
        import_submodules(package_name)

    predictor = None
    if args.predictor and args.archive_path:
        predictor = _get_predictor(args)

    app = make_app(predictor=predictor,
                    data_path=args.data_path,
                    static_dir=args.static_dir)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print(f"Model loaded, serving demo on port {args.port}")
    http_server.serve_forever()

if __name__ == "__main__":
    main(sys.argv[1:])