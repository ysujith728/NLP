# Natural Language Processing Lab

This repository contains the laboratory exercises, assignments, and implementations completed as part of the Natural Language Processing course.

New NLP experiments and lab activities will be added to this repository throughout the course.

## Lab Work

### Lab 1: NLTK Food Text Analysis

The first lab introduces basic Natural Language Processing techniques using the Natural Language Toolkit (NLTK) and Scikit-learn.

For this activity, a text document describing my favourite food, **Mutton Biryani**, was created and analyzed.

#### NLP Techniques Implemented

- Text file reading
- Text conversion to lowercase
- Punctuation removal
- Word tokenization using NLTK
- Word-frequency analysis using `Counter`
- English stop-word removal
- Word stemming using the Porter Stemmer
- TF-IDF vectorization using Scikit-learn
- Transformation and analysis of a new test sentence

#### Test Sentence

> Mutton biryani has delicious rice and spices.

The trained TF-IDF model identifies words from the original document. The Porter Stemmer converts some words into their stemmed forms, such as:

- `delicious` → `delici`
- `spices` → `spice`

## Repository Structure

```text
NLP/
│
├── README.md
│
└── NLTK-Food-Text-Analysis/
    │
    ├── lab1.ipynb
    ├── mutton_biryani.txt
    │
    ├── DOCS/
    │   └── LAB1.docx
    │
    └── Screenshots/
