from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# classifier = pipeline("sentiment-analysis")

# res = classifier("I hate this movie")

# print(res)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I hate this movie")

print(res)
