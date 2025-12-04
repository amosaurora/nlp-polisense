# run_training.py
from src.data_loader import load_and_split_data
from scripts.preprocess import tokenize_dataset
from src.model import get_model
from scripts.evaluate import compute_metrics, analyze_predictions
from scripts.train import training_args
from transformers import Trainer

# --- Load & Split Data ---
raw_datasets = load_and_split_data("data/bias_score_dataset/sentence_bias_dataset.json")

# --- Tokenize datasets ---
tokenized_datasets, tokenizer = tokenize_dataset(raw_datasets,"saved_models/roberta_bias")

# --- Initialize model ---
model = get_model("saved_models/roberta_bias")

# --- Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

# --- Train ---
trainer.train()

# --- Evaluate ---
metrics_summary, df_results = analyze_predictions(trainer, tokenized_datasets["test"], raw_datasets["test"])
print(metrics_summary)

save_dir = "saved_models/roberta_leaning_bias_result"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)