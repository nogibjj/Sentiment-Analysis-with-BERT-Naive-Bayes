from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AdamW
import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = "/imdb_master.csv"

# Import IMDB reviews
# synthetic_df = pd.read_csv(path, usecols=["review", "label"])


# Load the data into a Pandas dataframe
df = pd.read_csv(path, encoding="ISO-8859-1", usecols=["review", "label"])
df["label"].replace({"neg": 0, "pos": 1}, inplace=True)
train_set, test_set = train_test_split(df, test_size=0.2)


#########################################################################

# Split the data into input sequences and labels
text = train_set["review"].values
labels = list(train_set["label"].values)

# Tokenize the input sequences
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encoded_data_train = tokenizer.batch_encode_plus(
    text,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors="pt",
)


# Convert the inputs, labels, and attention masks to tensors

inputs = encoded_data_train["input_ids"]
mask = encoded_data_train["attention_mask"]
labels = torch.tensor(labels)


#################################################################
# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Specify the number of classes and maximum sequence length for the task
num_classes = 2
max_seq_length = 256


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

########################################################
# Set the hyperparameters for fine-tuning the BERT model
lr = 2e-5
batch_size = 32
epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Fine-tune the BERT model on the training data
model.train()


# Create a TensorDataset object with the training data
train_data = TensorDataset(inputs, labels, mask)


# Create a DataLoader object for the training data
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Specify the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=lr)

# Train the model
for _ in range(epochs):
    # Set the model to training mode
    model.train()

    # Iterate over the training data in batches
    for step, batch in enumerate(train_dataloader):
        # Unpack the inputs from the dataloader
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear any previously calculated gradients
        model.zero_grad()

        # Forward pass
        loss, logits = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )

        # Backward pass
        loss.backward()

#########################################################test
test_text = test_set["review"].values
test_labels = list(test_set["label"].values)

encoded_test = tokenizer.batch_encode_plus(
    test_text,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors="pt",
)

# Convert the inputs, labels, and attention masks to tensors

inputs_test = encoded_test["input_ids"]
mask_test = encoded_test["attention_mask"]
labels_test = torch.tensor(test_labels)


# Set the model to evaluation mode
model.eval()

# Predict the labels for the test data
with torch.no_grad():
    logits = model(inputs_test, token_type_ids=None, attention_mask=mask_test)


# Calculate the accuracy of the predictions
predicted_labels = [np.argmax(logit, dim=1) for logit in logits]
test_accuracy = accuracy_score(labels_test, predicted_labels)
