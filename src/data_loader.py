import json
import random
from datasets import Dataset, DatasetDict

def load_and_split_data(file_path, seed=42):
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found at {file_path}. Using fallback data.")
        data = [{'Text': f"Example sentence {i}", 'Score': random.choice([1,2,3,4])} for i in range(100)]

    random.seed(seed)
    random.shuffle(data)
    total_size = len(data)
    train_size = int(0.8 * total_size)
    validation_size = int(0.1 * total_size)

    train_data = Dataset.from_list(data[:train_size])
    validation_data = Dataset.from_list(data[train_size:train_size + validation_size])
    test_data = Dataset.from_list(data[train_size + validation_size:])

    return DatasetDict({
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    })
