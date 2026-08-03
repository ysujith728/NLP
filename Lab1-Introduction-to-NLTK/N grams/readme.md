# N-Grams using NLTK

This project demonstrates the implementation of **N-Grams** using the **Natural Language Toolkit (NLTK)** in Python. It covers the generation of **Unigrams, Bigrams, and Trigrams**, along with the implementation of a **Bigram Language Model** using **Laplace Smoothing**, **Sentence Probability**, and **Perplexity**.

---

## 📌 Objective

The objectives of this project are to:

- Understand the concept of N-Grams.
- Generate Unigrams, Bigrams, and Trigrams from a text corpus.
- Construct a Bigram Count Matrix.
- Apply Laplace (Add-One) Smoothing.
- Calculate Bigram Probabilities.
- Compute Sentence Probability.
- Evaluate the language model using Perplexity.

---

## 📂 Project Structure

```
N grams/
│── ngrams.py                  # Unigram, Bigram & Trigram generation
│── bigram_language_model.py   # Bigram language model implementation
│── README.md                  # Project documentation
```

---

## 🛠️ Technologies Used

- Python 3.x
- NLTK
- Pandas
- Collections (Counter)
- Math

---

## 📚 Concepts Covered

- Text Tokenization
- Vocabulary Construction
- Unigrams
- Bigrams
- Trigrams
- Bigram Count Matrix
- Bigram Frequency Analysis
- Laplace (Add-One) Smoothing
- Probability Matrix
- Sentence Probability
- Perplexity
- Language Modeling

---

# Program 1 – N-Gram Generation

## Description

This program demonstrates the extraction of **Unigrams**, **Bigrams**, and **Trigrams** from a small text corpus using NLTK.

### Corpus

```
Doc 1 : I like biryani.
Doc 2 : I like mutton.
Doc 3 : Chicken is tasty.
```

### Workflow

1. Tokenize the documents.
2. Convert words to lowercase.
3. Generate:
   - Unigrams
   - Bigrams
   - Trigrams
4. Compute frequency distribution.
5. Display the most common N-Grams.

### Example

Sentence

```
I like biryani
```

Unigrams

```
I
like
biryani
```

Bigrams

```
I like
like biryani
```

Trigrams

```
I like biryani
```

---

# Program 2 – Bigram Language Model

## Description

This program implements a complete **Bigram Language Model** with **Laplace Smoothing** and evaluates a test sentence using **Sentence Probability** and **Perplexity**.

### Training Documents

```
<s> I like biryani </s>

<s> I like chicken </s>

<s> chicken is tasty </s>
```

---

## Workflow

### Step 1

Tokenize every sentence.

```
<s>

I

like

biryani

</s>
```

---

### Step 2

Construct the vocabulary.

Example

```
<s>

I

like

biryani

chicken

is

tasty

</s>
```

---

### Step 3

Generate all bigrams.

Example

```
(<s>, I)

(I, like)

(like, biryani)

(biryani, </s>)
```

---

### Step 4

Build the Bigram Count Matrix.

Rows represent the current word.

Columns represent the next word.

Each cell stores the frequency of occurrence.

---

### Step 5

Generate Bigram Frequencies.

Example

```
(I, like) → 2

(like, chicken) → 1

(chicken, is) → 1
```

---

### Step 6

Apply Laplace (Add-One) Smoothing.

Formula

```
P(w₂|w₁)

=

Count(w₁,w₂)+1
-------------------------
Count(w₁)+Vocabulary Size
```

This ensures that unseen bigrams receive a non-zero probability.

---

### Step 7

Test Sentence

```
I like chicken
```

Sentence with tags

```
<s>

I

like

chicken

</s>
```

Generated Bigrams

```
(<s>, I)

(I, like)

(like, chicken)

(chicken, </s>)
```

---

### Step 8

Lookup Bigram Probabilities.

Each bigram probability is retrieved from the Laplace-smoothed probability matrix.

---

### Step 9

Sentence Probability

The probability of a sentence is calculated as the product of all bigram probabilities.

Formula

```
P(Sentence)

=

P(w₁|<s>)

×

P(w₂|w₁)

×

...

×

P(</s>|wₙ)
```

---

### Step 10

Perplexity

The language model is evaluated using Perplexity.

Formula

```
Perplexity

=

(1 / P(sentence))^(1/N)
```

where

- **P(sentence)** = Sentence Probability
- **N** = Number of Bigrams

Lower perplexity indicates a better language model.

---

### Step 11

Final Verdict

The program classifies the test sentence based on the calculated Perplexity.

Example

```
Sentence : I like chicken

Sentence Probability : 0.xxxxxxxx

Perplexity : x.xxxx

The sentence is highly probable according to the language model.
```

---

## Applications of N-Grams

N-Grams are widely used in:

- Language Modeling
- Text Prediction
- Autocomplete Systems
- Machine Translation
- Speech Recognition
- Spell Checking
- Chatbots
- Search Engines
- Sentiment Analysis
- Text Classification

---

## Learning Outcomes

After completing this project, you will be able to:

- Understand N-Gram models.
- Generate Unigrams, Bigrams, and Trigrams.
- Build a Bigram Language Model.
- Construct a Bigram Count Matrix.
- Apply Laplace Smoothing.
- Calculate Bigram Probabilities.
- Compute Sentence Probability.
- Evaluate a language model using Perplexity.

---

## References

- NLTK Documentation: https://www.nltk.org/
- NLTK `ngrams()` API: https://www.nltk.org/api/nltk.util.html

---
