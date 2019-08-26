from allennlp.models.archival import archive_model
import sys

archive_model(sys.argv[1], archive_path=sys.argv[2])