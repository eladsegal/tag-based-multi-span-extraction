import os
import json
from collections import OrderedDict

from src.preprocessing.augmentation.shuffle_persons_augmentation import ShufflePersonAugmentation
from src.preprocessing.utils import standardize_dataset

from tqdm import tqdm

if __name__ == "__main__":	
	# Load dataset    
    drop_path = os.path.join(os.path.join("data", "drop_dataset_train_clean.json"))

    with open(drop_path, 'r') as f:
        drop = json.load(f)
    drop = standardize_dataset(drop)

    # Define augmentation
    augmentation_ratio = 1
    augmentation = ShufflePersonAugmentation(augmentation_ratio=augmentation_ratio)
    augmented_drop = OrderedDict()

    # Perform augmentation
    augmented_counter = 0

    for passage_id, data in tqdm(drop.items()):
        for qa_pair in data['qa_pairs']:
            query_id = qa_pair['query_id']

            augmented_passages = augmentation.augment(passage_id, qa_pair['query_id'], data['passage'], qa_pair['answer']['spans'])
            
            if passage_id not in augmented_drop:
                augmented_drop[passage_id] = data

            if len(augmented_passages) > 0:
                qa_pair['query_id'] = f"AUG_{query_id}"

            for new_passage_text in augmented_passages:
                new_passage_id = f"{str(augmented_counter)}_AUG_{passage_id}"
                new_query_id = f"{str(augmented_counter)}_AUG_{query_id}"

                new_qa_pair = qa_pair.copy()
                new_qa_pair['query_id'] = new_query_id
                
                augmented_drop[new_passage_id] = OrderedDict()
                augmented_drop[new_passage_id]['passage'] = new_passage_text
                augmented_drop[new_passage_id]['qa_pairs'] = [new_qa_pair]

                augmented_counter += 1

    # Output the augmented dataset
    augmented_drop_path = os.path.join(os.path.join("data", "drop_dataset_train_augmented.json"))

    with open(augmented_drop_path, 'w') as f:
        json.dump(augmented_drop, f, indent=4)



