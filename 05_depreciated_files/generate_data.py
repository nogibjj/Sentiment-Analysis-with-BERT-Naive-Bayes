import numpy as np
import pandas as pd
from generative_example import get_real_data_preprocessed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import poisson
import random


def generate_data():
    """Get real data."""
    print("Getting real data")
    X, y = get_real_data_preprocessed()
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    X_v = vectorizer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_v, y, test_size=0.5, random_state=0, stratify=y
    )

    print("Get real data done")
    print("Generating reviews")
    # model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    num_reviews_each = 1000


    # decide length by "review" population
    mu = np.mean([len(x) for x in X])
    r = poisson.rvs(mu, size=num_reviews_each*2)  # set how many reviews we want

    # generate pos data
    pos_class_prob_sorted = model.feature_log_prob_[1, :].argsort()
    pos_class_prob = np.exp(model.feature_log_prob_[1, :])
    pos_class_prob = pos_class_prob / pos_class_prob.sum()
    pos_class_prob = pos_class_prob[pos_class_prob_sorted]

    # generate neg data
    neg_class_prob_sorted = model.feature_log_prob_[0, :].argsort()
    neg_class_prob = np.exp(model.feature_log_prob_[0, :])
    neg_class_prob = neg_class_prob / neg_class_prob.sum()
    neg_class_prob = neg_class_prob[neg_class_prob_sorted]
    
    Xg=[]
    s_pos=[]
    s_neg=[]

    vocab = vectorizer.get_feature_names()
    word_prob = np.exp(model.feature_log_prob_)
    # for i in range(r.shape[0]):
    for i in range(num_reviews_each):
        # pos=(np.take(vectorizer.get_feature_names(),np.random.choice(pos_class_prob_sorted,r[i], p=pos_class_prob)))
        pos1 = random.choices(vocab, word_prob[1], k = r[i])
        # stich the words together to form a sentence
        # switch g to list
        pos=list(pos1)
        # print(g)
        # print(type(g))
        s_pos=" ".join(pos)
        Xg.append(s_pos)
    yg = np.ones(num_reviews_each)

    pos_df = pd.DataFrame({"review":Xg, "label":yg})

    for i in range(num_reviews_each):
        # neg=(np.take(vectorizer.get_feature_names(),np.random.choice(neg_class_prob_sorted,r[i], p=neg_class_prob)))
        neg1 = random.choices(vocab, word_prob[0], k = r[i])

        # stich the words together to form a sentence
        # switch g to list
        neg=list(neg1)
        # print(g)
        # print(type(g))
        s_neg=" ".join(neg)
        Xg.append(s_neg)
    yg = np.append(yg, np.zeros(num_reviews_each))
   
    neg_df = pd.DataFrame({"review":Xg, "label":yg})

    whole_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # save the generated data
    whole_df.to_csv("generated_data.csv", index=False)
    # print(Xg)

    # yg = np.ones(r.shape[0])
    return Xg, yg

def main():
    Xg, yg = generate_data()
    vectorizer = CountVectorizer()
    vectorizer.fit(Xg)
    Xg = vectorizer.transform(Xg)
    X_train, X_test, y_train, y_test = train_test_split(
        Xg, yg, test_size=0.2, random_state=43)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print(f"Model socre on training dataset is {model.score(X_train, y_train)}")
    # print(f"Model f-1 socre on training dataset is {model.metrics.f1_score(X_train, y_train)}")

    print(f"Model socre on testing dataset is {model.score(X_test, y_test)}")
    # print(f"Model f-1 socre on training dataset is {model.metrics.f1_score(X_test, y_test)}")

    y_pred = model.predict(X_test)
    pred_accuracy = np.mean(y_pred == y_test)
    print(pred_accuracy)


if __name__ == "__main__":
    main()
