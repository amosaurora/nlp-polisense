import numpy as np
import pandas as pd

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = (preds == p.label_ids).mean()
    return {"accuracy": accuracy}

def analyze_predictions(trainer, tokenized_test, raw_test):
    # Run predictions
    predictions = trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    texts = [ex['Text'] for ex in raw_test]

    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': y_true == y_pred
    })

    # Compute differences
    df['diff'] = df['predicted_label'] - df['true_label']

    over_pred = df[df['diff'] > 0]
    under_pred = df[df['diff'] < 0]
    correct_pred = df[df['diff'] == 0]
    within_1 = df[np.abs(df['diff']) <= 1]

    # Compute test metrics
    test_loss = predictions.metrics.get("test_loss", None)
    test_accuracy = predictions.metrics.get("test_accuracy", (y_pred == y_true).mean())

    metrics_summary = {
        'total_samples': len(df),
        'over_predicted': len(over_pred),
        'under_predicted': len(under_pred),
        'correct': len(correct_pred),
        'within_1': len(within_1),
        'within_1_pct': len(within_1)/len(df),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }

    # Save predictions
    df.to_csv("outputs/test_predictions.csv", index=False)

    return metrics_summary, df