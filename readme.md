# Natural Language Processing Lab

This repository contains laboratory exercises, assignments, and practical implementations completed as part of the **Natural Language Processing (NLP)** course.

New NLP experiments, concepts, and lab activities will be added to this repository throughout the course.

## Lab Work

### Lab 1: NLTK Food Text Analysis

The first lab introduces fundamental Natural Language Processing techniques using the **Natural Language Toolkit (NLTK)** and **Scikit-learn**.

For this activity, a text document describing my favourite food, **Mutton Biryani**, was created and analyzed using different text-preprocessing and feature-extraction techniques.

#### NLP Techniques Implemented

* Text-file reading
* Text conversion to lowercase
* Punctuation removal
* Word tokenization using NLTK
* Word-frequency analysis using `Counter`
* English stop-word removal
* Word stemming using the Porter Stemmer
* TF-IDF vectorization using Scikit-learn
* Transformation and analysis of a new test sentence

#### Test Sentence

> Mutton biryani has delicious rice and spices.

The trained TF-IDF model identifies the words from the test sentence that are present in the vocabulary generated from the original document.

The Porter Stemmer converts some words into their corresponding stemmed forms, such as:

* `delicious` → `delici`
* `spices` → `spice`

#### Output Screenshots

The `Screenshots` folder contains the outputs generated during the experiment, including:

* Word-frequency analysis
* Stop-word removal
* Stemming output
* TF-IDF matrix
* Test-sentence sparse matrix
* Test-sentence TF-IDF scores

## Repository Structure

```text
NLP/
│
├── .gitignore
├── README.md
│
└── NLTK-Food-Text-Analysis/
    │
    ├── lab1.ipynb
    ├── mutton_biryani.txt
    │
    ├── Docs/
    │   └── LAB1.docx
    │
    └── Screenshots/
        ├── 01_word_frequency.png
        ├── 02_stopword_removal.png
        ├── 03_stemming_output.png
        ├── 04_tfidf_matrix.png
        ├── 05_test_sentence_sparse_matrix.png
        └── 06_test_sentence_tfidf_scores.png
```

## Technologies Used

* Python
* Jupyter Notebook
* NLTK
* Scikit-learn

## Future Updates

Additional NLP laboratory exercises, assignments, and implementations will be added as the course progresses.
