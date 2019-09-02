from allennlp.models.archival import archive_model
import sys

if len(sys.argv) < 4:
    archive_model(sys.argv[1], archive_path=sys.argv[2])
else:
    archive_model(sys.argv[1], weights = sys.argv[3], archive_path=sys.argv[2])