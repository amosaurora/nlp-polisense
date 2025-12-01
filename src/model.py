from transformers import RobertaForSequenceClassification
from src.config import config

num_labels = 3  # left / center / right
model = RobertaForSequenceClassification.from_pretrained(
    config.base_model, num_labels=num_labels
)
