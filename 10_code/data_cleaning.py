"""Data cleaning and preprocessing for IMDB reviews

NLP Final Project

IDS 703/ECE 684 2022 Fall

Team Members:
    - Lorna Aine
    - Song Young Oh
    - Dingkun Yang

Reference:
    - https://www.kaggle.com/utathya/imdb-review-dataset
    - https://www.nltk.org/

"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# install nltk data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")


def data_cleaning(file_path, pos_standard):

    """Data cleaning and preprocessing for IMDB reviews"""

    print("\nLoading data...")

    reviews_df = pd.read_csv(file_path, encoding="latin-1", usecols=["review", "label"])

    print("\nData loads successfully!")

    standards = pos_standard

    reviews_df["label"].replace(
        {"neg": 0, "pos": 1}, inplace=True
    )  # replace labels with 0 and 1

    print("\nCleaning data...")
    print("Removing URLs...")
    print("Removing numbers...")
    print("Tokenizing sentences...")
    print("Removing non-standard POS tags...")
    print("\nIt may take few minutes to run...")

    review_clean = []
    for review in reviews_df["review"]:
        tmp_review = re.sub(r"http\S+", "", review)  # remove urls
        tmp_review1 = re.sub(r"\d+", "", tmp_review)  # remove numbers
        sent_text = nltk.sent_tokenize(tmp_review1)  # tokenize sentences
        tmp = []
        for sentence in sent_text:
            tokenized_text = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokenized_text)
            for i in tagged:
                if i[1] in standards:
                    tmp.append(i[0])
        tmp1 = " ".join(tmp)
        review_clean.append(tmp1)
    reviews_clean_df = pd.DataFrame(
        {"review": review_clean, "label": reviews_df["label"]}
    )

    print("\nRemoving stop words...")
    print("Remove special characters...")
    print("Stemming words...")
    print("Removing HTML tags...")

    no_stops = []
    for review in reviews_clean_df["review"]:
        clean = re.sub(r"<.*?>", "", review)  # remove html tags
        clean1 = re.sub(r'[?|!|\'|"|#|>|<]', "", clean)  # remove special characters
        clean2 = re.sub(r"[.|,|)|(|\|/]", "", clean1)  # remove special characters
        stop_words = set(stopwords.words("english"))  # remove stop words
        sno = nltk.stem.SnowballStemmer("english")
        filtered_sentence = []
        word_tokens = word_tokenize(clean2)
        for word in word_tokens:
            if word not in stop_words:
                w_stemmed = sno.stem(word.lower())
                filtered_sentence.append(w_stemmed)
        clean_review = " ".join(filtered_sentence)
        no_stops.append(clean_review)
    reviews_no_stops_df = pd.DataFrame(
        {"review": no_stops, "label": reviews_clean_df["label"]}
    )

    print("\nData cleaning and preprocessing complete!")

    reviews_no_stops_df.to_csv("cleaned_reviews.csv", encoding="UTF-8", index=False)

    print("\nCleaned data saved to cleaned_reviews.csv")


if __name__ == "__main__":
    # Set Path to IMDb reviews
    FILE_PATH = "imdb_master.csv"
    # Define POS standards
    pos_tag = [
        "RB",
        "RBS",
        "RBR",
        "JJ",
        "JJR",
        "JJS",
        "VB",
        "VBD",
        "VBG",
        "VBP",
        "VBZ",
        "VBN",
    ]
    data_cleaning(file_path=FILE_PATH, pos_standard=pos_tag)
