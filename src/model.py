from transformers import RobertaForSequenceClassification

def get_model(model_checkpoint="roberta-base", num_labels=3):
    model = RobertaForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels
    )
    return model
