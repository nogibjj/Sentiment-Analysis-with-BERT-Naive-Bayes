import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import warnings
import nltk
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax


def clean_data(data_df, remove_stopwords=False):
    """Clean the data by removing URLs, converting to lowercase and removing @s and #s from the review"""

    warnings.filterwarnings("ignore")

    # replace nan with empty string
    data_df["review"] = data_df["review"].fillna("")

    # remove all URLs from the review
    data_df["review"] = data_df["review"].str.replace(r"http\S+", "")

    # remove all mentions from the review and replace with generic flag
    data_df["review"] = data_df["review"].str.replace(r"@\S+", "@user")

    # remove all hashtags from the review
    data_df["review"] = data_df["review"].str.replace(r"#", "")

    # lowercase all review
    data_df["review"] = data_df["review"].str.lower()

    if remove_stopwords:
        # remove stopwords
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        data_df["review"] = data_df["review"].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )
    return data_df


def load_model(task="sentiment-latest"):
    """Load the model and tokenizer for the task."""
    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment-latest

    warnings.filterwarnings("ignore")

    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # load the model
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    return config, tokenizer, model


def analyze_sentiment_score(review, config, tokenizer, model):
    # PT
    encoded_input = tokenizer(review, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ans = []
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    label = config.id2label[ranking[0]]
    if label == "Positive":
        return scores[ranking[0]]
    elif label == "Negative":
        return scores[ranking[0]] * -1
    else:
        return 0


def return_sentiment(review, config, tokenizer, model):
    """Return sentiment with highest polarity scores"""
    encoded_input = tokenizer(review, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ans = []
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return config.id2label[ranking[0]]


def apply_sentiment_to_df(df, config, tokenizer, model, calculate_scores):
    """Apply the sentiment analysis model to the dataframe."""
    print(df.columns)
    df["sentiment"] = df["review"].apply(
        lambda x: return_sentiment(x, config, tokenizer, model)
    )
    if calculate_scores == True:
        df["sentiment_score"] = df["review"].apply(
            lambda x: analyze_sentiment_score(x, config, tokenizer, model)
        )
    return df


def find_sentiment(df, keyword):
    """Find the sentiment of a given keyword. Assumes the dataframe has been cleaned and sentiment analysis has been applied."""
    keyword_review = df[df["review"].str.contains(keyword)]
    return keyword_review["sentiment"].value_counts(normalize=True)


def sentiment_generator(
    df, calculate_scores=False, task="sentiment-latest", remove_stopwords=False
):
    """Generate sentiment for each review in the dataframe.
    If calculate_scores is set to True, then the sentiment scores will be calculated."""

    # clean and preprocess data
    df = clean_data(df, remove_stopwords)

    # load model
    config, tokenizer, model = load_model(task)

    # apply sentiment analysis to dataframe
    df = apply_sentiment_to_df(df, config, tokenizer, model, calculate_scores)

    return df


if __name__ == "__main__":
    # load data
    df = pd.read_csv("../00_source_data/imdb_master.csv", encoding="unicode_escape")

    # generate sentiment
    df = sentiment_generator(df, calculate_scores=True)

    # save to csv
    df.to_csv("imdb_sentiment.csv")
