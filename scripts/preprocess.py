from transformers import RobertaTokenizerFast

def tokenize_dataset(raw_datasets, model_checkpoint="roberta-base", max_length=128):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["Text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        tokenized["labels"] = [score - 1 for score in examples["Score"]]
        return tokenized

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=['Text', 'Score']
    )
    return tokenized_datasets, tokenizer
