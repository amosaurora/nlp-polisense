from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    learning_rate=2e-5,
    remove_unused_columns=False,
)

training_args.logging_dir = "logs"
