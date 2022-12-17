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

    print("Ingestion of real data done")

    return X, y


def main():
    X, y = generate_data()
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4873
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
