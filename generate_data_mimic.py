import numpy as np
import pandas as pd

# from generative_example import get_real_data_preprocessed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import random


def generate_data():
    """Generate syntheic data with the distribution of real data."""
    print("Getting real data")
    # X, y = get_real_data_preprocessed()
    data = pd.read_csv("./cleaned_reviews.csv")
    X = data["review"].values.astype("U")
    y = data["label"]
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    X_v = vectorizer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_v, y, test_size=0.2, random_state=67575
    )

    print("Ingestion of real data done")

    # model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    num_reviews_each = 25000

    print(f"Generating {num_reviews_each} reviews for each class")

    # mimic the length by "review" population distribution
    data["length"] = data["review"].str.split().str.len()
    length = list(data["length"].values.astype("int"))

    Xg_pos = []
    Xg_neg = []
    s_pos = []
    s_neg = []
    Xg = []
    yg = []

    vocab = vectorizer.get_feature_names_out()
    word_prob = np.exp(model.feature_log_prob_)

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

    # save the generated data
    whole_df.to_csv("generated_data_mimic.csv", index=False)

    return Xg, yg


def main():
    Xg, yg = generate_data()
    vectorizer = CountVectorizer()
    vectorizer.fit(Xg)
    Xg = vectorizer.transform(Xg)
    X_train, X_test, y_train, y_test = train_test_split(
        Xg, yg, test_size=0.2, random_state=4873
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    acc_train = model.score(X_train, y_train) * 100
    acc_test = model.score(X_test, y_test) * 100

    print("\nThe Mean Accuracy:")
    print(f"Model score on training dataset is {acc_train:.2f}%")
    print(f"Model score on testing dataset is {acc_test:.2f}%")
    print("\nClassification Report :")
    print(classification_report(y_test, y_preds))


if __name__ == "__main__":
    main()


# k = 300
# all 1

# k = 30
# Model score on training dataset is 0.960425
# Model score on testing dataset is 0.9159
# 0.9159

# k = 5-15
# Model score on training dataset is 0.8657
# Model score on testing dataset is 0.7634
# 0.7634
