"""BERT Model with Hugging Face python Script

NLP Final Project

IDS 703/ECE 684 2022 Fall

Team Members:
    - Lorna Aine
    - Song Young Oh
    - Dingkun Yang
"""

from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def bert_model(file_path="synthetic_imdb.csv", model_huggingface="bert-base-uncased"):

    """Train the BERT model on the the file using the huggingface model"""

    path = file_path  # run naive_bayes_model.py first to get the synthetic data

    # Import IMDB reviews
    reviews_df = pd.read_csv(
        path,
        encoding="UTF-8",
        usecols=["review", "label"],
        dtype={"review": str, "label": int},
    )

    reviews_df = (
        reviews_df.dropna().copy()
    )  # drop reviews with null values, should be only 3

    # Make train and test data
    train_set, test_set = train_test_split(reviews_df, test_size=0.2)
    print("Train set size: ", train_set.shape[0])
    print("Test set size: ", test_set.shape[0])

    tds = Dataset.from_pandas(train_set, split="train")
    vds = Dataset.from_pandas(test_set, split="validation")

    ds = DatasetDict()
    ds["train"] = tds
    ds["validation"] = vds

    tokenizer = AutoTokenizer.from_pretrained(model_huggingface)

    def tokenize_function(examples):
        return tokenizer(examples["review"], padding="max_length", truncation=True)

    # tokenized_datasets = ds.map(tokenize_function, batched=True)

    TT_train_set = tds.map(tokenize_function, batched=True)
    TT_test_set = vds.map(tokenize_function, batched=True)

    model = BertForSequenceClassification.from_pretrained(
        model_huggingface, num_labels=2
    )

    training_args = TrainingArguments(output_dir="BERT_Result/Generated/")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir="BERT_Result/Generated/", evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TT_train_set,
        eval_dataset=TT_test_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(TT_test_set)

    pprint(predictions.metrics)


if __name__ == "__main__":

    # Original IMDb dataset
    # file_path = "imdb_master.csv"

    # Synthetic IMDb dataset
    file_path = "synthetic_imdb.csv"

    model = "bert-base-uncased"

    bert_model(file_path, model)
