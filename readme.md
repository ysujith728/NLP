# Natural Language Processing Lab

This repository contains the laboratory exercises, assignments, and implementations completed as part of the **Natural Language Processing** course.

New NLP experiments and lab activities will be added to this repository throughout the course.

## Lab Work

### Lab 1: NLTK Food Text Analysis

The first lab introduces fundamental Natural Language Processing techniques using the **Natural Language Toolkit (NLTK)** and **Scikit-learn**.

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

---

### Lab 2: TF-IDF Vectorization and Cosine Similarity

The second lab extends the text analysis performed in Lab 1 by representing documents numerically using **TF-IDF vectors** and measuring their similarity to a query using **cosine similarity**.

The same `mutton_biryani.txt` file is used for this activity. Each sentence in the text is treated as an individual document and labelled as `D1`, `D2`, `D3`, and so on.

#### NLP Techniques Implemented

- Sentence tokenization using NLTK
- Conversion of sentences into individual documents
- Text conversion to lowercase
- Punctuation removal
- Word tokenization
- English stop-word removal
- TF-IDF vocabulary generation
- TF-IDF vectorization of documents and query
- Document and query vector generation
- Cosine-similarity calculation
- Document ranking based on similarity
- Identification of the most similar document
- Dimensionality reduction using Principal Component Analysis (PCA)
- Two-dimensional visualization of TF-IDF vectors

#### Query

> delicious mutton biryani with rice and spices

#### Results

The Mutton Biryani text was divided into **9 documents**, with each sentence representing one document.

The TF-IDF model generated:

- **9 document vectors**
- **1 query vector**
- **62 vocabulary terms**
- A TF-IDF matrix with a shape of **10 × 62**

The cosine-similarity results showed that **D1** was the document most relevant to the query.

#### Most Similar Document

> **D1:** Mutton biryani is my favourite food because of its rich aroma and delicious taste.

**Cosine Similarity Score:** `0.370`

The TF-IDF vectors were reduced from 62 dimensions to two dimensions using **Principal Component Analysis (PCA)** and visualized in a two-dimensional scatter plot.

---

## Repository Structure

```text
NLP/
│
├── README.md
│
└── NLTK-Food-Text-Analysis/
    │
    ├── lab1.ipynb
    ├── lab2_tfidf_cosine_similarity.ipynb
    ├── mutton_biryani.txt
    │
    ├── Docs/
    │   └── LAB1.docx
    │
    └── Screenshots/
        │
        ├── Lab1/
        │   ├── 01_word_frequency.png
        │   ├── 02_stopword_removal.png
        │   ├── 03_stemming_output.png
        │   ├── 04_tfidf_matrix.png
        │   ├── 05_test_sentence_sparse_matrix.png
        │   └── 06_test_sentence_tfidf_scores.png
        │
        └── Lab2/
            ├── 01_documents.png
            ├── 02_processed_documents.png
            ├── 03_document_count.png
            ├── 04_tfidf_matrix_shape.png
            ├── 05_tfidf_matrix.png
            ├── 06_tfidf_scores.png
            ├── 07_vector_shapes.png
            ├── 08_cosine_similarity.png
            ├── 09_cosine_similarity_table.png
            ├── 10_ranked_similarity.png
            ├── 11_most_similar_document.png
            ├── 12_pca_components.png
            └── 13_tfidf_vectors_pca.png
```

## Technologies and Libraries

- Python
- Jupyter Notebook
- NLTK
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Future Work

Additional NLP laboratory experiments, implementations, documents, and visualizations will be added as the course progresses.