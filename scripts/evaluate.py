import numpy as np
import pandas as pd
import torch

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = (preds == p.label_ids).mean()
    return {"accuracy": accuracy}

def analyze_predictions(trainer, tokenized_test, raw_test):
    # Run predictions
    predictions = trainer.predict(tokenized_test)
    logits = predictions.predictions
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    texts = [ex['Text'] for ex in raw_test]
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    confidence = probs.max(axis=1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'true_label': y_true,
        'predicted_label': y_pred,
        'confidence': confidence,
        'correct': y_true == y_pred
    })

    # Compute differences
    df['diff'] = df['predicted_label'] - df['true_label']

    correct_pred = df[df['diff'] == 0]

    # Compute test metrics
    test_loss = predictions.metrics.get("test_loss", None)
    test_accuracy = predictions.metrics.get("test_accuracy", (y_pred == y_true).mean())

    metrics_summary = {
        'total_samples': len(df),
        'correct': len(correct_pred),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }

    # Save predictions
    df.to_csv("outputs/test_predictions.csv", index=False)

    return metrics_summary, df