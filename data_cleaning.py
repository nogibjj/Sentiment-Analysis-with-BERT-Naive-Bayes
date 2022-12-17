import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def data_cleaning(file_path, pos_standard):

    """Data cleaning and preprocessing for IMDB reviews"""

    reviews_df = pd.read_csv(file_path, encoding="latin-1", usecols=["review", "label"])

    standards = pos_standard

    reviews_df["label"].replace(
        {"neg": 0, "pos": 1}, inplace=True
    )  # replace labels with 0 and 1

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

    no_stops = []
    for review in reviews_clean_df["review"]:
        clean = re.sub(r"<.*?>", "", review)  # remove html tags
        clean1 = re.sub(r'[?|!|\'|"|#|>|<]', "", clean)  # remove special characters
        clean2 = re.sub(r"[.|,|)|(|\|/]", "", clean1)  # remove special characters
        # clean3 = re.sub(r"[.|,|)|(|\|/]", "", clean1)
        stop_words = set(stopwords.words("english"))  # remove stop words
        sno = nltk.stem.SnowballStemmer("english")
        filtered_sentence = []
        word_tokens = word_tokenize(clean2)
        for wd in word_tokens:
            if wd not in stop_words:
                w_stemmed = sno.stem(wd.lower())
                filtered_sentence.append(w_stemmed)
        clean_review = " ".join(filtered_sentence)
        no_stops.append(clean_review)
    reviews_no_stops_df = pd.DataFrame(
        {"review": no_stops, "label": reviews_clean_df["label"]}
    )
    reviews_no_stops_df.to_csv("cleaned_reviews.csv", encoding="UTF-8", index=False)


if __name__ == "__main__":
    # Import IMDB reviews
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
