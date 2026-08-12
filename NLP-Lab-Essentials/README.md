# 🔬 NLP Lab Essentials: Core Algorithms and Implementations

This directory contains a complete, production-grade suite of Jupyter notebooks for essential Natural Language Processing (NLP) laboratory experiments. It ranges from foundational text preprocessing and vector semantics to advanced sequence labeling models implemented both with industry libraries and from scratch.

---

## 📂 Folder Structure

```text
NLP-Lab-Essentials/
│
├── README.md      # This documentation file
├── requirements.txt                       # Python dependencies
├── NLP_Lab_Complete.ipynb                 # All labs compiled into a single workspace
│
├── 01_Preprocessing.ipynb                 # Lab 1: Text Preprocessing Pipeline
├── 02_Word_Embeddings.ipynb               # Lab 2: Word2Vec Word Semantics
├── 03_TFIDF_LSI.ipynb                     # Lab 3: Manual TF-IDF and Latent Semantic Indexing
├── 04_Ngrams.ipynb                        # Lab 4: N-gram Generation & Modeling
├── 05_HMM_POS_Tagging.ipynb               # Lab 5: HMM Sequence Labeling (hmmlearn)
├── 06_HMM_Viterbi_From_Scratch.ipynb      # Lab 6: Dynamic Programming Viterbi Decoder from Scratch
└── 07_CRF_POS_Tagging.ipynb               # Lab 7: CRF Sequence Labeling (sklearn-crfsuite)
```

---

## 🔬 Lab Walkthroughs and Technical Specifications

### 1. [Lab 1: Preprocessing Pipeline (01_Preprocessing.ipynb)](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/01_Preprocessing.ipynb)
*   **Goal**: Transform raw text data into clean, normalized tokens suitable for downstream machine learning tasks.
*   **Concepts & Operations**:
    *   **File I/O**: Loads text from a corpus file [sample.txt](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/data/sample.txt).
    *   **Case Normalization**: Standardizes all characters to lowercase to prevent vocabulary fragmentation (e.g., treating `Mutton` and `mutton` as distinct).
    *   **Punctuation Removal**: Uses translation tables to strip out punctuation marks:
        ```python
        text.translate(str.maketrans("", "", string.punctuation))
        ```
    *   **Tokenization**: Splits continuous text into word-level strings using NLTK's `word_tokenize`.
    *   **Stop Word Filtering**: Evaluates tokens against the NLTK stop word corpus to eliminate high-frequency, low-semantic words (e.g., `is`, `the`, `with`).
    *   **Stemming**: Truncates words to their base lexical root (e.g., `tasty` $\rightarrow$ `tasti`, `grilled` $\rightarrow$ `grill`) using the Porter Stemmer algorithm.
*   **Dependencies**: `nltk`, `string`

---

### 2. [Lab 2: Word Semantics & Embeddings (02_Word_Embeddings.ipynb)](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/02_Word_Embeddings.ipynb)
*   **Goal**: Train word embeddings from a text corpus using a Skip-gram architecture to represent words in a dense vector space.
*   **Key Parameters & Settings**:
    *   `vector_size=50`: Output dimension size for the dense embeddings.
    *   `window=2`: Context window checking 2 words to the left and right.
    *   `min_count=1`: Considers all words, including single occurrences.
    *   `sg=1`: Selects the **Skip-gram** architecture (predicts surrounding context words given a target word) instead of CBOW (`sg=0`).
*   **Operations**:
    *   **Model Training**: Trains Gensim's `Word2Vec` model on a tokenized food text corpus.
    *   **Vector Retrieval**: Accesses the underlying 50-dimensional weights for individual tokens.
    *   **Semantic Similarity**: Computes the Cosine Similarity between word pairs (e.g., `chicken` and `curry`).
    *   **Neighbor Retrieval**: Extracts the top $k$ closest words in vector space to a query word using spatial proximity.
*   **Dependencies**: `gensim`

---

### 3. [Lab 3: Custom TF-IDF and Latent Semantic Indexing (03_TFIDF_LSI.ipynb)](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/03_TFIDF_LSI.ipynb)
*   **Goal**: Implement a TF-IDF matrix calculator from scratch and perform Latent Semantic Indexing (LSI) to capture latent topics.
*   **Mathematical Formulations**:
    *   **Term Frequency (TF)**:
        $$\text{TF}(t, d) = \frac{\text{Count}(t, d)}{\sum_{t' \in d} \text{Count}(t', d)}$$
    *   **Inverse Document Frequency (IDF)**:
        $$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
        Where $N$ is the total documents and $\text{DF}(t)$ is the number of documents containing term $t$.
    *   **Dimensionality Reduction (LSI/SVD)**:
        Applies Singular Value Decomposition (SVD) to decompose the TF-IDF matrix $X$ into:
        $$X \approx U \Sigma V^T$$
        Using scikit-learn's `TruncatedSVD` to project documents into a lower $k$-dimensional latent topic space.
*   **Code Flow**:
    1. Parse 5 raw documents $\rightarrow$ extract unique vocabulary.
    2. Build manual Term Frequency (TF) matrix using `pandas`.
    3. Calculate Inverse Document Frequency (IDF) table.
    4. Compute final $\text{TF-IDF} = \text{TF} \times \text{IDF}$ matrix.
    5. Fit `TruncatedSVD(n_components=2)` to reduce document-term space into 2 component coordinates.
*   **Dependencies**: `pandas`, `numpy`, `math`, `scikit-learn`

---

### 4. [Lab 4: N-gram Frequencies and Language Modeling (04_Ngrams.ipynb)](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/04_Ngrams.ipynb)
*   **Goal**: Generate, count, and analyze word-level N-gram structures both manually and using machine learning vectorizers.
*   **Methodologies**:
    *   **Manual NLTK Approach**: Tokenizes sentences and generates contiguous sequences (Unigrams, Bigrams, Trigrams) using NLTK `ngrams`. Uses python's `Counter` to measure frequencies.
    *   **Scikit-Learn Count Vectorizer**: Uses `CountVectorizer(ngram_range=(1, 2))` to build a sparse document-term matrix representing both individual words and bigram pairs.
*   **Dependencies**: `nltk`, `collections`, `scikit-learn`

---

### 5. [Lab 5: POS Tagging with HMM Libraries (05_HMM_POS_Tagging.ipynb)](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/05_HMM_POS_Tagging.ipynb)
*   **Goal**: Perform Part-of-Speech (POS) tagging using a Hidden Markov Model (HMM) package.
*   **Model Components**:
    *   **Hidden States ($S$)**: POS tags `NN` (Noun), `VB` (Verb), `RB` (Adverb).
    *   **Observations ($O$)**: Vocabulary words (`dogs`, `cats`, `run`, `chase`, `quickly`, `slowly`).
    *   **Start Probabilities ($\pi$)**: Likelihood of starting a sentence with tag $s$.
    *   **Transition Probability Matrix ($A$)**: Probability of moving from tag $s_t$ to tag $s_{t+1}$ ($A_{ij} = P(s_j \mid s_i)$).
    *   **Emission Probability Matrix ($B$)**: Probability of observing word $w$ given tag $s$ ($B_{ik} = P(w_k \mid s_i)$).
*   **Implementation**: Configures a `CategoricalHMM` instance from `hmmlearn.hmm`, maps state matrices, and tags the test sentence `"dogs run quickly"`.
*   **Dependencies**: `numpy`, `hmmlearn`

---

### 6. [Lab 6: Viterbi Decoding Algorithm from Scratch (06_HMM_Viterbi_From_Scratch.ipynb)](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/06_HMM_Viterbi_From_Scratch.ipynb)
*   **Goal**: Implement the Viterbi dynamic programming decoding algorithm from scratch to evaluate HMM POS tagging paths without external libraries.
*   **Viterbi Recurrence Math**:
    *   **Initialization** (Time $t = 0$):
        $$V_{0, s} = \ln(\pi_s) + \ln(P(w_0 \mid s))$$
    *   **Recursion** (Time $t$ from 1 to $T-1$):
        $$V_{t, s} = \max_{s'} \left( V_{t-1, s'} + \ln(A_{s', s}) \right) + \ln(P(w_t \mid s))$$
        $$\text{bp}_{t, s} = \arg\max_{s'} \left( V_{t-1, s'} + \ln(A_{s', s}) \right)$$
    *   **Termination** (Time $T-1$):
        $$\text{best\_path\_prob} = \max_{s} V_{T-1, s}$$
        $$\text{last\_state} = \arg\max_{s} V_{T-1, s}$$
    *   **Backtracking**: Traverse backwards from $t = T-1$ to $t = 0$ using backpointers $\text{bp}_{t, s}$ to retrieve the most likely state sequence.
*   **Implementation**: Implements this log-probability algorithm in pure Python to process sentence inputs and print clean tag mappings.
*   **Dependencies**: `math`

---

### 7. [Lab 7: Feature-based POS Tagging using CRF (07_CRF_POS_Tagging.ipynb)](file:///d:/5TH%20SEMESTER/NLP/NLP-Lab-Essentials/07_CRF_POS_Tagging.ipynb)
*   **Goal**: Train a Conditional Random Field (CRF) sequence tagger and evaluate its performance.
*   **Features Extracted per Word**:
    *   Bias element (`bias = 1.0`).
    *   Lowercase form of the target word.
    *   Character-level suffixes (`word[-3:]`, `word[-2:]`) and prefixes (`word[:3]`, `word[:2]`).
    *   Formatting boolean attributes (`word.isupper()`, `word.istitle()`, `word.isdigit()`).
    *   Contextual neighborhood checks: previous word properties ($i-1$) and next word properties ($i+1$).
    *   Sequence boundaries: `BOS` (Beginning of Sentence) and `EOS` (End of Sentence) flags.
*   **Evaluation**: Uses `sklearn-crfsuite` to fit the CRF model, predicts tags for unseen sequences, and prints precision, recall, F1, and accuracy tables.
*   **Dependencies**: `sklearn-crfsuite`, `scikit-learn`

---

## 🚀 Setup and Execution Instructions

### 1. Install Dependencies
Run the pip installer on the folder's requirement file to retrieve all necessary machine learning libraries:
```bash
pip install -r requirements.txt
```

### 2. Run Jupyter Notebooks
Start the local Jupyter server to explore, run, and modify the experiments:
```bash
jupyter notebook
```
Once the interface loads, you can run either the individual lab notebooks (`01_Preprocessing.ipynb` to `07_CRF_POS_Tagging.ipynb`) or open the compiled master workspace (`NLP_Lab_Complete.ipynb`).
