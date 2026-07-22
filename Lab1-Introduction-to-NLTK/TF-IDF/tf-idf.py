import nltk
import pandas as pd
import string
import math

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Documents
documents = [
    "I like mutton biryani.",
    "Chicken curry tastes good with mutton fry.",
    "I enjoy grilled chicken.",
    "Fish fry is tasty.",
    "I am a non vegetarian foodie."
]

# Preprocessing
stop_words = set(stopwords.words("english"))

processed_docs = []

for doc in documents:
    doc = doc.lower()
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(doc)
    tokens = [w for w in tokens if w not in stop_words]
    processed_docs.append(tokens)

print("Processed Documents\n")
for i, doc in enumerate(processed_docs):
    print(f"D{i+1}: {' '.join(doc)}")

# Vocabulary
vocab = sorted(set(word for doc in processed_docs for word in doc))

print("\nVocabulary:\n")
print(vocab)

# TF TABLE

tf_table = pd.DataFrame(
    0.0,
    index=vocab,
    columns=[f"D{i+1}" for i in range(len(processed_docs))]
)

for i, doc in enumerate(processed_docs):
    total = len(doc)

    for word in vocab:
        tf = doc.count(word) / total
        tf_table.iloc[vocab.index(word), i] = round(tf, 3)

print("\n====================")
print("TERM FREQUENCY (TF)")
print("====================\n")

print(tf_table)

# IDF TABLE

N = len(processed_docs)

idf_values = {}

for word in vocab:
    df = sum(word in doc for doc in processed_docs)
    idf = math.log(N / df)
    idf_values[word] = round(idf, 3)

idf_table = pd.DataFrame.from_dict(
    idf_values,
    orient="index",
    columns=["IDF"]
)

print("\n====================")
print("IDF TABLE")
print("====================\n")

print(idf_table)

# TF-IDF TABLE

tfidf_table = tf_table.copy()

for word in vocab:
    tfidf_table.loc[word] = (
        tf_table.loc[word] * idf_values[word]
    ).round(3)

print("\n====================")
print("TF-IDF = TF × IDF")
print("====================\n")

print(tfidf_table)