import json
import random
from datasets import Dataset, DatasetDict
import accelerate
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import wandb 

# --- Data Loading and Splitting (Your Existing Code) ---
relative_path = '../data/sora/sentence_score.json'

try:
    with open(relative_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: File not found at {relative_path}. Using fallback data.")
    # Ensure fallback data has scores 1-4
    data = [{'Text': f"Example sentence {i}", 'Score': random.choice([1, 2, 3, 4])} for i in range(100)] 

random.shuffle(data)

total_size = len(data)
train_size = int(0.8 * total_size)
validation_size = int(0.1 * total_size)

train_data = data[:train_size]
validation_data = data[train_size:train_size + validation_size]
test_data = data[train_size + validation_size:]

train_dataset = Dataset.from_list(train_data)
validation_dataset = Dataset.from_list(validation_data)
test_dataset = Dataset.from_list(test_data)

raw_datasets = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})


# --- 1. Load Tokenizer and Define Constants ---

MODEL_CHECKPOINT = "roberta-base"
# 🎯 FIX 1: Set to 4 (for zero-indexed labels 0, 1, 2, 3)
NUM_LABELS = 4 

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

# --- 2. Preprocess (Tokenize) the Data ---

def tokenize_and_align_labels(examples):
    # Tokenize the input text
    tokenized_inputs = tokenizer(
        examples["Text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    # 🎯 FIX 2: Map scores 1-4 to labels 0-3
    tokenized_inputs["labels"] = [score - 1 for score in examples["Score"]] 
    return tokenized_inputs

# Apply the preprocessing function to all splits
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels, 
    batched=True,
    remove_columns=['Text', 'Score']
)

# --- 3. Define and Initialize Trainer ---

# Define the model.
# 🎯 FIX 3: Removed 'ignore_index=0' which caused the TypeError
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=NUM_LABELS
)

# Define a simple function to compute metrics
def compute_metrics(p):
    # Predictions and labels are now 0, 1, 2, or 3.
    preds = np.argmax(p.predictions, axis=1)
    accuracy = (preds == p.label_ids).mean()
    return {"accuracy": accuracy}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    learning_rate=2e-5,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=["wandb"], 
    run_name="roberta-bias-score-v1-labels0-3", # Updated run name
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

print("\n" + "=" * 50)
print("✅ Trainer Initialized. Using zero-indexed labels 0-3.")
print("=" * 50)
print(f"Model: {MODEL_CHECKPOINT}")
print(f"Number of training examples: {len(tokenized_datasets['train'])}")

# 1. Start Training
print("Starting RoBERTa Fine-tuning...")
train_results = trainer.train()

# 2. Display Training Metrics
print("\n--- Training Summary ---")
print(train_results.metrics)

# 3. Final Evaluation on the Test Set
print("\nRunning final evaluation on the Test Dataset...")
test_results = trainer.predict(tokenized_datasets["test"])

# Display the test metrics
print("\n--- Test Set Metrics ---")
print(test_results.metrics)
print("-" * 25)


# import pandas as pd
# import numpy as np

# # Run predictions on the test dataset
# predictions = trainer.predict(tokenized_datasets["test"])

# # Convert logits to predicted class indices
# y_pred = np.argmax(predictions.predictions, axis=1)

# # True labels
# y_true = predictions.label_ids

# # Optional: get original text back from the raw dataset
# texts = [example['Text'] for example in test_dataset]

# # Combine into a DataFrame
# df_results = pd.DataFrame({
#     'text': texts,
#     'true_label': y_true,
#     'predicted_label': y_pred,
#     'correct': y_true == y_pred
# })

# # Print first 20 results
# print(df_results.head(20))

# # Optionally, save full results to CSV
# df_results.to_csv("test_predictions.csv", index=False)
# print("\n✅ Test predictions saved to 'test_predictions.csv'")
