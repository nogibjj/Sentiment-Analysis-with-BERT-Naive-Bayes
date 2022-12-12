"""Generative model example."""
import numpy as np
from data import get_imdb_data
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


def cleanhtml(sentence):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, " ", sentence)
    return cleantext


def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r"", sentence)
    cleaned = re.sub(r"[.|,|)|(|\|/]", r" ", cleaned)
    return cleaned


def posneg(x):
    if x == "pos":
        return 1
    else:
        return 0


def get_real_data_preprocessed():
    """Get preprocessed IMDB data."""
    data = get_imdb_data()
    stop = set(stopwords.words("english"))
    sno = nltk.stem.SnowballStemmer("english")
    i = 0
    str1 = " "
    final_string = []
    all_positive_words = []
    all_negative_words = []
    s = ""
    for sent in data["review"].values:
        filtered_sentence = []
        sent = cleanhtml(sent)
        for w in sent.split():
            for cleaned_words in cleanpunc(w).split():
                if (cleaned_words.isalpha()) & (len(cleaned_words) > 2):
                    if cleaned_words.lower() not in stop:
                        s = (sno.stem(cleaned_words.lower())).encode("utf8")
                        filtered_sentence.append(s)
                        if (data["label"].values)[i] == "pos":
                            all_positive_words.append(s)
                        if (data["label"].values)[i] == "neg":
                            all_negative_words.append(s)
                    else:
                        continue
                else:
                    continue

        str1 = b" ".join(filtered_sentence)
        final_string.append(str1)
        i += 1
    data["cleaned_review"] = final_string
    filtered_score = data["label"].map(posneg)
    data["score"] = filtered_score
    X = np.array(data["cleaned_review"].values)
    y = np.array(data["score"].values)
    return X, y


def main():
    """Run the experiment."""
    X, y = get_real_data_preprocessed()
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0, stratify=y
    )
    model = BernoulliNB()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))


if __name__ == "__main__":
    main()
