import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import datasets
from datasets import Dataset, DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np

import evaluate
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# path = "generated_data.csv"
path = "generated_data_mimic.csv"


# Import IMDB reviews
# reviews_df = pd.read_csv(path, encoding="ISO-8859-1", usecols=["review", "label"])
# read in as int
reviews_df = pd.read_csv(
    path,
    encoding="UTF-8",
    usecols=["review", "label"],
    dtype={"review": str, "label": int},
)
# reviews_df = pd.read_csv(path, encoding="UTF-8", usecols=["review", "label"], )
# reviews_df["label"].replace({"neg": 0, "pos": 1}, inplace=True)

reviews_df = reviews_df.dropna()

# Make train and test data
train_set, test_set = train_test_split(reviews_df, test_size=0.2)
print("Train set size: ", train_set.shape[0])
print("Test set size: ", test_set.shape[0])


tds = Dataset.from_pandas(train_set, split="train")
vds = Dataset.from_pandas(test_set, split="validation")

ds = DatasetDict()
ds["train"] = tds
ds["validation"] = vds


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["review"], padding="max_length", truncation=True)


tokenized_datasets = ds.map(tokenize_function, batched=True)

TT_train_set = tds.map(tokenize_function, batched=True)
TT_test_set = vds.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)


training_args = TrainingArguments(output_dir="BERT_Result/Generated")


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     labels, predictions, average="binary"
    # )
    # acc = accuracy_score(labels, predictions)
    # return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir="BERT_Result/Generated_Cauchy", evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=TT_train_set,
    eval_dataset=TT_test_set,
    compute_metrics=compute_metrics,
)
trainer.train()

trainer.predict(TT_test_set)
