# Natural Language Processing Lab

This repository contains the laboratory exercises, assignments, and implementations completed as part of the **Natural Language Processing** course.

---

## 📂 Repository Structure

The project workspace is structured into specialized modules focusing on different aspects of Natural Language Processing:

```text
NLP/
│
├── readme.md                                  # Repository documentation
├── tfidf_table.csv                            # Exported TF-IDF table from scratch implementation
│
├── Lab1-Introduction-to-NLTK/                 # Introduction to NLTK and NLP foundations
│   ├── myself.txt                             # Profile text corpus used for Exercise 1
│   ├── food.txt                               # Food text corpus used for Exercise 2
│   ├── exercise1.ipynb                        # Word/Sentence Tokenization & Stop Words
│   ├── exercise2.ipynb                        # Step-by-step demonstration of the 5 NLP Phases
│   │
│   ├── Docs/
│   │   └── Lab1-Introduction-to-NLTK.docx     # Exercise documentation report
│   │
│   ├── N grams/
│   │   ├── ngrams_types.py                    # Unigram, Bigram, and Trigram extraction
│   │   └── ngram.py                           # Bigram Language Model, Laplace Smoothing & Perplexity
│   │
│   └── TF-IDF/
│       └── tf-idf.py                          # Math-based TF-IDF calculation from scratch
│
├── NLP-Lab-Essentials/                        # Essential NLP Laboratory Notebooks & Solutions
│   ├── README.md                              # Subfolder README with detailed lab walkthroughs
│   ├── requirements.txt                       # Package dependencies
│   ├── 01_Preprocessing.ipynb                 # File reading, tokenization, stopword removal, stemming
│   ├── 02_Word_Embeddings.ipynb               # Skip-Gram Word2Vec training and word similarities
│   ├── 03_TFIDF_LSI.ipynb                     # Manual TF-IDF and Latent Semantic Indexing (SVD)
│   ├── 04_Ngrams.ipynb                        # Manual & Scikit-learn N-gram generation
│   ├── 05_HMM_POS_Tagging.ipynb               # POS tagging with HMM using hmmlearn CategoricalHMM
│   ├── 06_HMM_Viterbi_From_Scratch.ipynb      # Step-by-step Viterbi algorithm from scratch
│   ├── 07_CRF_POS_Tagging.ipynb               # Feature-based POS Tagging using CRF (sklearn-crfsuite)
│   ├── NLP_Lab_Complete.ipynb                 # Compiled notebook with all experiments
│   └── data/
│       └── sample.txt                         # Input text corpus for Preprocessing
│
└── NLTK-Food-Text-Analysis/                   # Advanced Food Corpus Text Processing & Retrieval
    ├── mutton_biryani.txt                     # Text document describing Mutton Biryani
    ├── lab1.ipynb                             # Text preprocessing, stemming, and scikit-learn TF-IDF
    ├── lab2_tfidf_cosine_similarity.ipynb    # Document retrieval, Cosine Similarity, and PCA visualization
    │
    ├── Docs/
    │   └── LAB1.docx                          # Food Text Analysis report
    │
    └── Screenshots/                           # Step-by-step program execution outputs
        ├── Lab1/                              # Screenshots for text preprocessing & TF-IDF
        └── Lab2/                              # Screenshots for Cosine Similarity & PCA plot
```

---

## 🔬 Lab Modules and Implementations

### 1. Lab 1: Introduction to NLTK (`Lab1-Introduction-to-NLTK/`)

This directory covers foundational NLP concepts ranging from text preprocessing to language models and mathematical representations.

#### 📝 [Exercise 1: Tokenization and Stop Words](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/exercise1.ipynb)
Demonstrates initial lexical processing on a personal description corpus ([myself.txt](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/myself.txt)):
- **Word Tokenization**: Segmenting raw strings into word tokens.
- **Stop Words Identification**: Identifying and filtering standard English stop words using the NLTK corpus.
- **Sentence Filtering**: Filtering stop words from individual sentences and cleaning up punctuation spacing.

#### 📝 [Exercise 2: Demonstration of the 5 Phases of NLP](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/exercise2.ipynb)
Uses [food.txt](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/food.txt) to illustrate the end-to-end NLP workflow:
1. **Lexical Analysis**: Breaking text into paragraphs, sentences, and words.
2. **Syntactic Analysis**: Annotating parts of speech (POS Tagging) on tokens.
3. **Semantic Analysis**: Extracting word meanings/definitions using WordNet and identifying entities with Named Entity Recognition (NER).
4. **Discourse Integration**: Coreference resolution tracking (e.g., mapping pronouns like `its` to their real-world subject).
5. **Pragmatic Analysis**: Determining contextual sentiment/attitude using the NLTK VADER sentiment analyzer.

#### 🔠 N-gram Generation & Modeling
Located in the [N grams/](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/N%20grams) subdirectory:
- **[ngrams_types.py](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/N%20grams/ngrams_types.py)**: Extracts and computes frequency distributions for **Unigrams**, **Bigrams**, and **Trigrams** over a simple sentence corpus.
- **[ngram.py](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/N%20grams/ngram.py)**: Implements a **Bigram Language Model** from scratch:
  - Vocabulary indexing and special start/end tags (`<s>`, `</s>`).
  - Bigram Frequency Matrix building using `pandas`.
  - **Laplace (Add-One) Smoothing** to resolve zero probability occurrences.
  - Sentence probability evaluation and **Perplexity** calculation to evaluate the model's fluency/likelihood on test sequences.

> [!TIP]
> **Perplexity** measures how well a probability distribution or language model predicts a sample. Lower perplexity indicates that the language model is highly confident and that the test sentence is natural/probable.

#### 🧮 Custom TF-IDF from Scratch
Located in the [TF-IDF/](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/TF-IDF) subdirectory:
- **[tf-idf.py](file:///d:/5TH%20SEMESTER/NLP/Lab1-Introduction-to-NLTK/TF-IDF/tf-idf.py)**: Performs mathematical TF-IDF calculations without relying on scikit-learn helper classes:
  - Custom calculation of **Term Frequency (TF)**:
    $$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total terms in } d}$$
  - Custom calculation of **Inverse Document Frequency (IDF)** using natural logarithm:
    $$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
  - Computation of the combined **TF-IDF** matrix:
    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
  - Saves the resulting vectors to [tfidf_table.csv](file:///d:/5TH%20SEMESTER/NLP/tfidf_table.csv) in the workspace root.

---

### 2. Food Text Analysis & Information Retrieval (`NLTK-Food-Text-Analysis/`)

This directory contains experiments using larger corpora and information retrieval metrics.

#### 📝 [Lab 1: NLTK Food Text Analysis](file:///d:/5TH%20SEMESTER/NLP/NLTK-Food-Text-Analysis/lab1.ipynb)
An introductory activity using a text description of a favorite food ([mutton_biryani.txt](file:///d:/5TH%20SEMESTER/NLP/NLTK-Food-Text-Analysis/mutton_biryani.txt)):
- Normalization (lowercase conversion, punctuation removal).
- Stemming using the NLTK **Porter Stemmer** (e.g., `delicious` → `delici`, `spices` → `spice`).
- TF-IDF Vectorization using **scikit-learn**'s `TfidfVectorizer`.
- Vectorizing and examining a sample test sentence: *"Mutton biryani has delicious rice and spices."*

#### 📝 [Lab 2: TF-IDF Vectorization and Cosine Similarity](file:///d:/5TH%20SEMESTER/NLP/NLTK-Food-Text-Analysis/lab2_tfidf_cosine_similarity.ipynb)
Extends the food text analysis to document retrieval:
- **Document Segmentation**: Treating each sentence of the `mutton_biryani.txt` file as an independent document (yielding 9 documents).
- **Query Processing**: Vectorizing a custom query: *"delicious mutton biryani with rice and spices"*.
- **Similarity Scoring**: Computing the **Cosine Similarity** between the query vector and all document vectors:
  $$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
- **Document Ranking**: Sorting documents by score. The highest relevance score of `0.370` belongs to document **D1**: *"Mutton biryani is my favourite food because of its rich aroma and delicious taste."*
- **Dimensionality Reduction & Visualization**: Reducing the 62-dimensional TF-IDF vectors into a 2D space using **Principal Component Analysis (PCA)** and plotting them.

---

### 3. Essential NLP Labs (`NLP-Lab-Essentials/`)

This folder contains a set of fundamental NLP lab notebooks implementing core workflows from basic text processing to advanced sequence classification:

*   **[01_Preprocessing.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/01_Preprocessing.ipynb)**: Implements text tokenization, casing normalization, punctuation removal, stopword filtering, and Porter stemming using `NLTK`. Reads from [sample.txt](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/data/sample.txt).
*   **[02_Word_Embeddings.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/02_Word_Embeddings.ipynb)**: Demonstrates training a Skip-gram model using Gensim's `Word2Vec`, vector queries, and cosine similarities on a vocabulary corpus.
*   **[03_TFIDF_LSI.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/03_TFIDF_LSI.ipynb)**: Calculates TF, IDF, and TF-IDF from scratch using pandas and math libraries, and reduces dimensionality with Latent Semantic Indexing (LSI) via scikit-learn's `TruncatedSVD`.
*   **[04_Ngrams.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/04_Ngrams.ipynb)**: Generates character/word n-grams (unigrams, bigrams, trigrams) manually with NLTK `ngrams` and automatically via scikit-learn's `CountVectorizer`.
*   **[05_HMM_POS_Tagging.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/05_HMM_POS_Tagging.ipynb)**: Performs POS Tagging using a manually defined Hidden Markov Model (HMM) from the `hmmlearn` package.
*   **[06_HMM_Viterbi_From_Scratch.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/06_HMM_Viterbi_From_Scratch.ipynb)**: Implements the Viterbi dynamic programming decoding algorithm from scratch for POS tagging sequence evaluation.
*   **[07_CRF_POS_Tagging.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/07_CRF_POS_Tagging.ipynb)**: Employs Conditional Random Fields (CRFs) using `sklearn-crfsuite` with detailed word feature extraction rules.
*   **[NLP_Lab_Complete.ipynb](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/NLP_Lab_Complete.ipynb)**: Compilation of all these seven modules in a single Jupyter Notebook.

---

## 🛠️ Technologies and Libraries

- **Programming Language**: Python 3.10+
- **Environments**: Jupyter Notebooks
- **Key Libraries**:
  - `NLTK` (Tokenization, POS tagging, Stemming, WordNet, VADER sentiment, N-grams)
  - `Scikit-learn` (TfidfVectorizer, PCA, TruncatedSVD)
  - `Pandas` (Matrix construction, CSV manipulation, tabular formatting)
  - `NumPy` (Vector and numeric computations)
  - `Matplotlib` (PCA scatter plotting)
  - `Gensim` (Word2Vec word embeddings)
  - `hmmlearn` (HMM-based sequence classification)
  - `sklearn-crfsuite` (Conditional Random Fields model implementation)

---

## 🚀 How to Run the Scripts

### Prerequisites
Make sure Python and the required libraries are installed. You can install all dependencies using the requirements file in the essentials folder:
```bash
pip install -r "NLP-Lab-Essentials/requirements.txt"
```

### Running N-grams and Scratch TF-IDF
Navigate to the folders and run the Python scripts:
```bash
# To run N-grams modeling
python "Lab1-Introduction-to-NLTK/N grams/ngram.py"

# To compute custom TF-IDF matrices
python "Lab1-Introduction-to-NLTK/TF-IDF/tf-idf.py"
```

### Running the Essential Notebooks
Start the Jupyter environment and open the notebooks in `NLP-Lab-Essentials/`:
```bash
# Start Jupyter
jupyter notebook
```