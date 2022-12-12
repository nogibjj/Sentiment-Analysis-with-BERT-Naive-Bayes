#!/usr/bin/env python

"""
Fine Tuning Example with HuggingFace

Based on official tutorial
"""

from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

# Load the dataset
dataset = load_dataset("imdb")
# dataset["train"][100]
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# can use if needed to reduce memory usage and training time
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))


training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()  # train the model

trainer.push_to_hub()  # push the model to huggingface hub
tokenizer.push_to_hub(
    repo_id="fine_tune_imdb_distilbert"
)  # push the tokenizer to huggingface hub
