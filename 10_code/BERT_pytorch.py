
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#get data 
import os.path
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path,"./imdb_master.csv")
#f = open(path)


#Import IMDB reviews
reviews_df = pd.read_csv(path, encoding="ISO-8859-1", usecols=["review", "label"])


#Make train and test data 
train_set, test_set = train_test_split(reviews_df, test_size=0.2)
print("Train set size: ", train_set.shape[0])
print("Test set size: ", test_set.shape[0])
train_set.head()
test_set.head()

#bring in BERT
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)

#encode inputs
import torch


encoded_data_train = tokenizer.batch_encode_plus(
    train_set["review"],
    add_special_tokens=True,
    return_attention_mask=True,
    padding="longest",
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    test_set["review"],
    add_special_tokens=True,
    return_attention_mask=True,
    padding="longest",
    max_length=256,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_set["label"].values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(test_set["label"].values)

dataset_train = TensorDataset(input_ids_train, 
                              attention_masks_train,
                              labels_train)

dataset_val = TensorDataset(input_ids_val, 
                            attention_masks_val,
                           labels_val)

dataset_val.tensors