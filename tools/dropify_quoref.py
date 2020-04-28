import argparse
import os
import json

def main(args):
    with open(args.quoref_src, encoding = 'utf8') as input_file:
        dataset = json.load(input_file)

    dropified_dataset = {}
    paragraph_counter = 0
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            passage_info = dropified_dataset[paragraph['context_id'] if 'context_id' in paragraph else str(paragraph_counter)] = {}
            passage_info['passage'] = paragraph['context']
            if 'url' in article:
                passage_info['wiki_url'] = article['url']
            if 'title' in article:
                passage_info['title'] = article['title']

            qa_pairs = []
            passage_info['qa_pairs'] = qa_pairs
            for qa in paragraph['qas']:
                qa_pair = {}
                qa_pairs.append(qa_pair)

                qa_pair['question'] = qa['question']
                qa_pair['query_id'] = qa['id']

                if 'answers' in qa:
                    qa_pair['answer'] = {
                        'number': "",
                        'date': {
                            'day': "",
                            'month': "",
                            'date': ""
                        },
                        'spans': [answer['text'] for answer in qa['answers']]
                    }
                    qa_pair['original_answer'] = qa['answers']
            paragraph_counter += 1

    with open(args.dropified_quoref_dest, mode='w', encoding = 'utf8') as output_file:
        json.dump(dropified_dataset, output_file, indent=4)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("quoref_src", type=str, help="Source quoref file")
    parse.add_argument("dropified_quoref_dest", type=str, help="Output path")
    args = parse.parse_args()

    main(args)
