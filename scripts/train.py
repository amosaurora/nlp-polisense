from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=int(0.1 * 5 * 2588 / 8),
    weight_decay=0.05,
    logging_dir="logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=["wandb"],
    run_name="roberta-bias-score-small",
    remove_unused_columns=False,
)

training_args.logging_dir = "logs"