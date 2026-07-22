import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from collections import Counter
import math

# Download punkt tokenizer (Run only once)
nltk.download('punkt')
nltk.download('punkt_tab')

# =====================================================
# DOCUMENTS
# =====================================================

documents = [
    "I like biryani",
    "I like chicken",
    "chicken is tasty"
]

print("="*70)
print("DOCUMENTS")
print("="*70)

for i, doc in enumerate(documents, start=1):
    print(f"Doc {i}: <s> {doc} </s>")

# =====================================================
# TOKENIZATION
# =====================================================

tokenized_docs = []

for doc in documents:
    tokens = ["<s>"] + word_tokenize(doc) + ["</s>"]
    tokenized_docs.append(tokens)

print("\n" + "="*70)
print("TOKENIZED DOCUMENTS")
print("="*70)

for i, tokens in enumerate(tokenized_docs, start=1):
    print(f"D{i}: {tokens}")

# =====================================================
# VOCABULARY
# =====================================================

vocab = []

for tokens in tokenized_docs:
    for word in tokens:
        if word not in vocab:
            vocab.append(word)

print("\n" + "="*70)
print("VOCABULARY")
print("="*70)

print(vocab)
print("\nVocabulary Size =", len(vocab))

# =====================================================
# BIGRAMS
# =====================================================

all_bigrams = []

print("\n" + "="*70)
print("BIGRAMS")
print("="*70)

for i, tokens in enumerate(tokenized_docs, start=1):
    bg = list(bigrams(tokens))
    print(f"\nDoc {i}")
    for b in bg:
        print(b)
    all_bigrams.extend(bg)

# =====================================================
# BIGRAM COUNT MATRIX
# =====================================================

matrix = pd.DataFrame(0, index=vocab, columns=vocab)

for w1, w2 in all_bigrams:
    matrix.loc[w1, w2] += 1

# =====================================================
# ROW TOTALS
# =====================================================

matrix["Row Total"] = matrix.sum(axis=1)

print("\n" + "="*70)
print("BIGRAM COUNT MATRIX")
print("="*70)

print(matrix)

# =====================================================
# SHOW BIGRAM FREQUENCIES
# =====================================================

freq = Counter(all_bigrams)

print("\n" + "="*70)
print("BIGRAM FREQUENCIES")
print("="*70)

freq_df = pd.DataFrame({
    "Bigram": [str(k) for k in freq.keys()],
    "Frequency": list(freq.values())
})

print(freq_df)

# =====================================================
# LAPLACE SMOOTHING
# =====================================================

vocab_size = len(vocab)

# .astype(float) avoids a pandas dtype error when assigning
# float probabilities into what starts as an int64 matrix
prob_matrix = matrix.drop(columns=["Row Total"]).astype(float).copy()

for row in vocab:
    row_total = matrix.loc[row, "Row Total"]
    for col in vocab:
        count = matrix.loc[row, col]
        probability = (count + 1) / (row_total + vocab_size)
        prob_matrix.loc[row, col] = round(probability, 3)

print("\n" + "=" * 70)
print("LAPLACE SMOOTHED PROBABILITY MATRIX")
print("=" * 70)

print(prob_matrix)

# =====================================================
# TEST SENTENCE
# =====================================================

test_sentence = "I like chicken"

tokens = ["<s>"] + word_tokenize(test_sentence) + ["</s>"]
test_bigrams = list(bigrams(tokens))

print("\n" + "=" * 70)
print("TEST SENTENCE")
print("=" * 70)

print(test_sentence)
print("\nSentence with Tags:")
print(tokens)
print("\nBigrams:")
for bg in test_bigrams:
    print(bg)

# =====================================================
# LOOKUP PROBABILITIES
# =====================================================

lookup = []
sentence_probability = 1

for w1, w2 in test_bigrams:
    probability = prob_matrix.loc[w1, w2]
    lookup.append([w1, w2, probability])
    sentence_probability *= probability

lookup_df = pd.DataFrame(
    lookup,
    columns=["Current Word", "Next Word", "Probability"]
)

print("\n" + "=" * 70)
print("LOOKUP PROBABILITIES")
print("=" * 70)

print(lookup_df)

# =====================================================
# SENTENCE PROBABILITY
# =====================================================

print("\n" + "=" * 70)
print("SENTENCE PROBABILITY")
print("=" * 70)

print("P(Sentence) =", round(sentence_probability, 8))

# =====================================================
# PERPLEXITY
# =====================================================

N = len(test_bigrams)
perplexity = (1 / sentence_probability) ** (1 / N)

print("\n" + "=" * 70)
print("PERPLEXITY")
print("=" * 70)

print(round(perplexity, 4))

# =====================================================
# FINAL VERDICT
# =====================================================

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

print("Sentence :", test_sentence)
print("Sentence Probability :", round(sentence_probability, 8))
print("Perplexity :", round(perplexity, 4))

if perplexity < 10:
    print("\nThe sentence is highly probable according to the language model.")
else:
    print("\nThe sentence is less probable according to the language model.")