from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I hate this movie")

print(res)
