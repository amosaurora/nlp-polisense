from datasets import Dataset
import pandas as pd
import os
import json
from src.tokenizer_utils import tokenizer

DATA_DIR = os.path.join(
    os.getcwd(), "data/external/ramybaly-Article-Bias-Prediction/data/"
)
TSV_DIR = os.path.join(DATA_DIR, "splits/random")
JSON_DIR = os.path.join(DATA_DIR, "jsons")


def preprocess(batch_dataset):
    preprocessed_dataset = batch_dataset.map(
        tokenizer.tokenize, batched=False, remove_columns=["text"]
    )
    return preprocessed_dataset


def load_tsv_with_json_text(split):
    # Load TSV
    df = pd.read_csv(os.path.join(TSV_DIR, f"{split}.tsv"), sep="\t", header=0)
    df.rename(columns={"ID": "text", "bias": "label"}, inplace=True)
    df["text"] = df["text"].apply(
        lambda x: json.load(open(os.path.join(JSON_DIR, f"{x}.json")))
    )
    df["text"] = df["text"].apply(lambda x: f"{x['title']} {x['content']}")

    return Dataset.from_pandas(df)


preprocess(load_tsv_with_json_text("test"))
