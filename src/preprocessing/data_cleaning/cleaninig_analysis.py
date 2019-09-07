import os 
import json

cleaning_logs_dir = os.path.join(os.path.dirname(__file__), 'cleaning_logs')

questions = list()

for log_path in os.listdir(cleaning_logs_dir):
    with open(os.path.join(cleaning_logs_dir, log_path),'r') as f:
        cleaning_log = json.load(f)
    
    for passage_id, data in cleaning_log.items():
        questions.extend(data.keys())

print (len(set(questions)))
