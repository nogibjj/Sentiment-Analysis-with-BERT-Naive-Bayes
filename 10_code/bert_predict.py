"""BERT Prediction python Script with Pretrained Model Loaded"""

from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BertForSequenceClassification,
)


def bert_predict(
    file_path="synthetic_imdb.csv", model_path="BERT_Result/Generated/checkpoint-15000"
):

    """Predict the sentiment of the reviews in the given path using the pretrained model in the given model_path"""

    model = BertForSequenceClassification.from_pretrained(model_path)

    path = file_path

    # Import IMDB reviews

    reviews_df = pd.read_csv(
        path, encoding="UTF-8", usecols=["review", "label"], dtype={"label": int}
    )

    reviews_df = reviews_df.dropna().copy()

    # Make train and test data
    train_set, test_set = train_test_split(reviews_df, test_size=0.2)
    print("Train set size: ", train_set.shape[0])
    print("Test set size: ", test_set.shape[0])

    tds = Dataset.from_pandas(train_set, split="train")
    vds = Dataset.from_pandas(test_set, split="validation")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["review"], padding="max_length", truncation=True)

    TT_train_set = tds.map(tokenize_function, batched=True)
    TT_test_set = vds.map(tokenize_function, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir="BERT_Result/Generated/",
        evaluation_strategy="epoch",
        do_predict=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TT_train_set,
        eval_dataset=TT_test_set,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(TT_test_set)
    pprint(predictions.metrics)


if __name__ == "__main__":

    # Original IMDb dataset
    # file_path = "imdb_master.csv"
    # model_path = "BERT_Result/Real/checkpoint-15000" # Pretrained model on original IMDb dataset

    # Synthetic IMDb dataset
    file_path = "synthetic_imdb.csv"
    model_path = "BERT_Result/Generated/checkpoint-15000"  # Pretrained model on synthetic dataset

    bert_predict(file_path, model_path)
