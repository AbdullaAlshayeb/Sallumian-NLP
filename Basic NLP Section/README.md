
---

# 🧠 Basic NLP: Feature Engineering & Text Vectorization

This repository is a comprehensive technical suite focused on the **Classical NLP Pipeline**. It documents the transition from unstructured strings to high-dimensional numerical vectors, covering the essential preprocessing steps required before feeding text into Machine Learning models.

## 🔬 Notebooks & Technical Deep-Dive

### 1. Lexical Cleaning & Tokenization

* **`text_cleaning.ipynb`**: Implementation of noise reduction layers using Regex. Handles the removal of ASCII artifacts, HTML entities, and URL sanitization.
* **`tokenization.ipynb`**: Comparative analysis of **Word-level** vs. **Sentence-level** segmentation. Explores the boundary detection challenges in rule-based (NLTK) vs. statistical (SpaCy) engines.

### 2. Morphological Normalization

* **`stemming.ipynb`**: Exploration of the **Porter** and **Snowball** algorithms. Evaluates the computational speed of suffix-stripping versus the risk of **Over-stemming** (loss of semantic root) and **Under-stemming** (failure to link related words).
* **`lemmatization.ipynb`**: Utilizing WordNet and POS (Part-of-Speech) tagging to return words to their dictionary form (Lemma). This ensures "ran," "run," and "running" map to a single feature index.

### 3. Dimensionality Reduction

* **`stopwords.ipynb`**: Strategies for pruning high-frequency, low-entropy words. Discusses the trade-off between reducing feature space "noise" and preserving critical context for Sentiment Analysis (e.g., the importance of "not").

### 4. Vector Space Modeling

* **`bag_of_words.ipynb`**: Building a **Sparse Matrix** of term frequencies. Demonstrates the *Distributional Hypothesis*—that words appearing in similar contexts share meanings—while highlighting the "Lost Order" limitation.
* **`TF_IDF.ipynb`**: Mathematical implementation of the $TF \times IDF$ weight.

$$TF\text{-}IDF(t, d) = tf(t, d) \cdot \log\left(\frac{N}{df(t)}\right)$$



This notebook proves how we can statistically down-weight "corpus-wide" commonalities to amplify **discriminative features**.
* **`ngrams.ipynb`**: Preserving local sequence information by generating contiguous sequences of **n** items. Crucial for capturing phrases (e.g., "New York") that Unigrams would decompose incorrectly.

### 5. Advanced Pipeline & EDA

* **`preprocessing_pipeline.ipynb`**: A modular, reusable class-based approach to automate the entire flow, ensuring consistency between training data and real-world inference.
* **`text_augmentation.ipynb`**: Synthetic data generation using **Synonym Replacement**, **Random Deletion**, and **Back-translation** to combat class imbalance in small datasets.
* **`text_analysis.ipynb`**: Exploratory Data Analysis (EDA) focusing on **Zipf’s Law**, lexical diversity scores, and word-cloud distributions to identify underlying dataset biases.

---

## 📈 Next Steps: Semantic & Contextual Learning

The techniques in this "Basic" folder treat words as **discrete symbols**. The next phase of this project will bridge the gap between "counting" and "understanding" using Deep Learning.

### 🚀 Phase 2: Static Word Embeddings

* **Word2Vec (Mikolov et al.):** Moving from sparse to **dense vectors**. Implementing Skip-gram and CBOW architectures where words are mapped into a continuous vector space based on their neighbors.
* **GloVe & FastText:** Exploring global co-occurrence statistics and sub-word (character n-gram) information to handle **Out-of-Vocabulary (OOV)** tokens.

### 🚀 Phase 3: Contextual & Sequential Models

* **Recurrent Neural Networks (RNNs/LSTMs):** Capturing the temporal nature of language to solve long-term dependencies.
* **Transformers (BERT/GPT):** Implementing **Self-Attention** mechanisms. Understanding how the representation of a word like "bank" changes dynamically based on whether the context is "river" or "finance."

---

## 🛠️ Stack & Dependencies

* **Language:** Python 3.10+
* **Core NLP:** `NLTK`, `SpaCy`, `Scikit-Learn`
* **Data Science:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`

---
