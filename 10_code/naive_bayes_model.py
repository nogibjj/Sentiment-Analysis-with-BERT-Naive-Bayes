""" Naive Bayes Model and Synthetic Data Generation

NLP Final Project

IDS 703/ECE 684 2022 Fall

Team Members:
    - Lorna Aine
    - Song Young Oh
    - Dingkun Yang

Reference:
    - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    - https://scikit-learn.org/
"""

import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def count_vectorizer(X):
    """Count vectorizer."""
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)
    return X, vectorizer


def naive_bayes(X, y):
    """Naive Bayes model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=67575
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc_train = model.score(X_train, y_train) * 100
    acc_test = model.score(X_test, y_test) * 100

    # Model performance
    print("\nModel performance :")
    print("\nThe Mean Accuracy:")
    print(f"Model score on training dataset is {acc_train:.2f}%")
    print(f"Model score on testing dataset is {acc_test:.2f}%")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    return model


def generate_data(file="cleaned_reviews.csv"):

    """Generate syntheic data with the distribution of real data."""

    print("\nGetting real processed data")
    data = pd.read_csv(file)
    X = data["review"].values.astype("U")
    y = data["label"]

    X, vectorizer = count_vectorizer(X)

    print("\nIngestion of real data done")

    print("\n_______________Naive Bayes Model on real IMDb Data_______________")

    model = naive_bayes(X, y)

    num_reviews_each = 25000

    print(
        f"_______________Generating {num_reviews_each} reviews for each class_______________"
    )
    print("\nIt may take a few minutes...")

    # mimic the length by "review" population distribution
    data["length"] = data["review"].str.split().str.len()
    length = list(data["length"].values.astype("int"))

    Xg_pos = []
    Xg_neg = []
    s_pos = []
    s_neg = []
    Xg = []
    yg = []

    vocab = vectorizer.get_feature_names_out()  # get the vocabulary
    word_prob = np.exp(model.feature_log_prob_)  # get the probability of each word

    for i in range(num_reviews_each):
        # for postitive words
        pos_word = random.choices(vocab, word_prob[1], k=length[25000 + i])
        # stich the words together to form a sentence
        s_pos = " ".join(pos_word)
        Xg_pos.append(s_pos)
        # for negative words
        neg_word = random.choices(vocab, word_prob[0], k=length[i])
        # stich the words together to form a sentence
        s_neg = " ".join(neg_word)
        Xg_neg.append(s_neg)

    yg_pos = np.ones(num_reviews_each)
    pos_df = pd.DataFrame({"review": Xg_pos, "label": yg_pos})
    yg_neg = np.zeros(num_reviews_each)
    neg_df = pd.DataFrame({"review": Xg_neg, "label": yg_neg})

    Xg = np.append(Xg_pos, Xg_neg)
    yg = np.append(yg_pos, yg_neg)

    whole_df = pd.concat([pos_df, neg_df], ignore_index=True)

    print("\nSynthetic data generated successfully!")

    whole_df.to_csv(
        "synthetic_imdb.csv", index=False
    )  # save the generated data as csv file
    print("\nSynthetic data saved as synthetic_imdb.csv")
    return Xg, yg


def synthetic_data_result(file="cleaned_reviews.csv"):

    """Naive Bayes model on synthetic data."""

    Xg, yg = generate_data(file)  # generate synthetic data
    Xg = count_vectorizer(Xg)[0]  # vectorize the generated data

    print("\n____________Naive Bayes model on Synthetic Data____________")

    naive_bayes(Xg, yg)


if __name__ == "__main__":
    original_data = (
        "cleaned_reviews.csv"  # run data_cleaning.py first to get the cleaned data
    )
    synthetic_data_result(original_data)
