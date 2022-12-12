from transformers import pipeline
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

classifiers = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# classifiers = pipeline("sentiment-analysis")
url = "https://github.com/Yer1k/imdb_dataset/raw/main/imdb_master.csv"
df = pd.read_csv(url, encoding="unicode_escape")
# df = pd.read_csv("../00_source_data/imdb_master.csv", encoding="unicode_escape")
# df["sentiment"] = df["review"].apply(lambda x: classifiers(x)[0]["label"])
# df["sentiment_score"] = df["review"].apply(lambda x: classifiers(x)[0]["score"])

# data cleaning
# stop = set(stopwords.words('english'))
# sno = nltk.stem.SnowballStemmer('english')

# def cleanhtml(sentence):
#     cleanr = re.compile('<.*?>')
#     cleantext = re.sub(cleanr, ' ', sentence)
#     return cleantext

# def cleanpunc(sentence):
#     cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
#     cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
#     return cleaned

df["sentiment"] = ""
df["sentiment_score"] = 0.0
for i in range(len(df)):
    try:
        input = df["review"][i][0:1000]
        res = classifiers(input)
        df["sentiment"][i] = res[0]["label"]
        df["sentiment_score"][i] = res[0]["score"]
    except Exception as ee:
        print(i)
        print(df["review"][i])
        print(ee)
        break

    print(f"{i/len(df) *100}% done")
# # save to csv
df.to_csv("../01_processed_data/imdb_master_sentiment_codespaces.csv", index=False)

# res = classifier("I hate this movie")
# print(res)
