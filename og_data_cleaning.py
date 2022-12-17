import numpy as np
import pandas as pd
from data import get_imdb_data
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


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
    data = pd.read_csv("./imdb_master.csv", encoding="latin-1", nrows=50)
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
    data.to_csv("og_cleaned_reviews.csv", encoding="UTF-8", index=False)


if __name__ == "__main__":
    get_real_data_preprocessed()
