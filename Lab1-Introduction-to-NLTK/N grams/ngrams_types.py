"""
Lab Assignment: N-gram Generation

Description:
Demonstrates the extraction of Unigrams, Bigrams, and Trigrams
from a text corpus using NLTK.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist

# Ensure necessary NLTK resources are available
nltk.download('punkt', quiet=True)


def preprocess_document(document):
    """Tokenises and normalises a single document."""
    return [word.lower() for word in word_tokenize(document) if word.isalnum()]


def generate_and_display_ngrams(corpus, n, ngram_name):
    """Generates N-grams for the corpus and displays their frequencies."""

    print(f"--- {ngram_name} (N={n}) ---")

    all_ngrams = []

    for i, doc in enumerate(corpus):
        tokens = preprocess_document(doc)
        doc_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(doc_ngrams)
        print(f"Doc {i+1}: {doc_ngrams}")

    # Calculate and display overall frequencies
    freq_dist = FreqDist(all_ngrams)

    print(f"\nTop 5 Most Common {ngram_name}:")

    for ngram_tuple, count in freq_dist.most_common(5):
        ngram_str = " ".join(ngram_tuple)
        print(f"'{ngram_str}': {count}")

    print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":

    # Define the corpus
    documents = [
        "I like biryani.",
        "I like mutton.",
        "Chicken is tasty."
    ]

    print("Corpus:")

    for i, doc in enumerate(documents):
        print(f"Doc {i+1}: {doc}")

    print("\nConclusion: I am a 'Non-Vegetarian' foodie.\n")

    print("=" * 40 + "\n")

    # Generate Unigrams (N=1)
    generate_and_display_ngrams(documents, 1, "Unigrams")

    # Generate Bigrams (N=2)
    generate_and_display_ngrams(documents, 2, "Bigrams")

    # Generate Trigrams (N=3)
    generate_and_display_ngrams(documents, 3, "Trigrams")