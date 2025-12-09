import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = (preds == p.label_ids).mean()
    f1 = f1_score(p.label_ids, preds, average='macro')
    return {"accuracy": accuracy, "f1": f1}

def analyze_predictions(trainer, tokenized_test, raw_test, class_names=["Unbiased","Biased","No Agreement"]):
    predictions = trainer.predict(tokenized_test)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=1)
    y_true = predictions.label_ids
    texts = [ex['Text'] for ex in raw_test]

    if isinstance(logits, np.ndarray):
        logits_tensor = torch.tensor(logits)
    else:
        logits_tensor = logits

    probs = torch.softmax(logits_tensor, dim=1).numpy()
    confidence = probs.max(axis=1)

    df = pd.DataFrame({
        'text': texts,
        'true_label': y_true,
        'predicted_label': y_pred,
        'confidence': confidence,
        'correct': y_true == y_pred
    })
    df['diff'] = df['predicted_label'] - df['true_label']
    correct_pred = df[df['diff'] == 0]

    test_loss = predictions.metrics.get("test_loss", None) if predictions.metrics else None
    test_accuracy = predictions.metrics.get("test_accuracy", (y_pred == y_true).mean()) if predictions.metrics else (y_pred == y_true).mean()
    f1 = f1_score(y_true, y_pred, average='macro')

    metrics_summary = {
        'total_samples': len(df),
        'correct': len(correct_pred),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'f1': f1
    }

    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    total_preds = len(y_pred)
    print("Prediction counts per class:")
    for i, count in enumerate(pred_counts):
        if i < len(class_names):
            percentage = (count / total_preds) * 100 if total_preds > 0 else 0
            print(f"Class {class_names[i]}: {count} times ({percentage:.2f}%)")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='viridis',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14}
    )

    ax.xaxis.set_ticks_position('top')
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Sentence Leaning Transfer Learning, Sentence Bias', y=1.1, fontsize=14)
    plt.tight_layout()
    plt.show()

    return metrics_summary, df
