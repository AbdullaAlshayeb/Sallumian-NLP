---

# 🚀 Intermediate NLP: Semantic Representations & Predictive Modeling

This section marks the transition from classical, rule-based text processing to **Statistical Learning** and **Neural Embeddings**. The focus here is on capturing the **Latent Semantics** of language—mathematically representing the fact that "buy" and "purchase" occupy similar coordinates in a vector space despite sharing no character-level similarity.

---

## 📂 Detailed Notebook Analysis

### 1. Vector Semantics & Word Representations
* **`01_word_embeddings.ipynb`**: Implementation of **Word2Vec** architectures (**CBOW & Skip-gram**). This notebook explores the transition from high-dimensional, sparse One-Hot Encodings to low-dimensional, dense vectors. We analyze how a shallow neural network learns word associations by predicting context, allowing for vector arithmetic (e.g., $vec("King") - vec("Man") + vec("Woman") \approx vec("Queen")$).
    
* **`02_document_embeddings.ipynb`**: Scaling embeddings from individual tokens to entire sequences using **Doc2Vec (Paragraph Vectors)**. We contrast simple **Averaging/Pooling** strategies with the **Distributed Memory (PV-DM)** approach, which treats a Document ID as a "memory" token that provides global context to the model during training.

### 2. Supervised Learning & Sentiment Analysis
* **`03_text_classification_models.ipynb`**: A comparative study of discriminative classifiers. We implement **Naive Bayes** (probabilistic), **Logistic Regression** (linear), and **Support Vector Machines** (geometric margin maximization). This notebook focuses on the trade-offs between model interpretability and boundary complexity in high-dimensional text data.
* **`04_sentiment_analysis.ipynb`**: A specialized application of classification focused on human emotion. We address the **Negation Problem** (e.g., "not good") by implementing **N-grams** to capture local word order and utilizing **TF-IDF weighting** to suppress uninformative stop words while highlighting sentiment-dense tokens.

### 3. Information Extraction & Unsupervised Learning
* **`05_named_entity_recognition.ipynb`**: Extracting structured data from unstructured text using **SpaCy’s Statistical Pipelines**. We examine the **BIO (Begin, Inside, Outside) Tagging** format and how transition probabilities allow a model to identify multi-word entities like "New York City" as a single `GPE` (Geopolitical Entity).
* **`06_topic_modeling.ipynb`**: Discovering hidden thematic structures using **Latent Dirichlet Allocation (LDA)**. This notebook treats documents as a mixture of topics and topics as a mixture of words, utilizing **Dirichlet Distributions** to cluster documents without pre-defined labels.

### 4. Similarity & Feature Engineering
* **`07_text_similarity.ipynb`**: Building search and recommendation logic. We implement **Cosine Similarity** to measure the angular distance between document vectors, demonstrating how embeddings can power "semantic search" that outperforms keyword-based matching.
* **`08_feature_engineering.ipynb`**: Enhancing model performance by extracting meta-features beyond word counts. We calculate **Part-of-Speech (POS) distributions**, **Lexical Diversity (TTR)**, and **Readability Indices** (Flesch-Kincaid) to provide models with structural hints about the text.

---

## ✅ Technical Accomplishments
* **Dimensionality Reduction:** Successfully compressed feature spaces from $10,000+$ dimensions to $\approx 300$, preserving semantic integrity while reducing computational overhead.
* **Metric-Driven Evaluation:** Moved beyond Accuracy to utilize **F1-Scores, Precision-Recall Curves, and Confusion Matrices**, ensuring model reliability even in the presence of severe class imbalance.
* **Inference Capabilities:** Implemented logic to transform brand-new, unseen text into the existing vector space for real-time prediction.

---

## 🔮 The Road Ahead: Advanced Deep Learning for NLP

With the statistical foundations solid, the next phase of this repository focuses on **Neural Architectures** and **Contextual Language Models**.

1.  **`01_neural_networks_basics`**: Building the computational graphs (Backpropagation, Activation Functions) that serve as the engine for all modern NLP.
2.  **`02_embedding_layers`**: Learning how to train embeddings *inside* a model specifically for a task, rather than using pre-trained static vectors.
3.  **`03_rnn_lstm`**: Addressing the "Sequence Problem" using **Recurrent Neural Networks** and **Long Short-Term Memory** units to handle long-range dependencies in text.
    
4.  **`04_attention`**: Implementing the **Attention Mechanism** to allow models to "focus" on specific parts of a sentence regardless of distance.
5.  **`05_transformers`**: Building the **Encoder-Decoder** architecture from scratch, moving away from recurrence toward parallelized self-attention.
6.  **`06_pretrained_models`**: Leveraging SOTA models like **BERT and GPT-2** to jumpstart performance using Transfer Learning.
7.  **`07_fine_tuning_bert`**: Customizing a massive language model for specific downstream tasks like Document Classification or NER.

---

### 🛠️ Environment Requirements
```bash
pip install gensim spacy scikit-learn seaborn matplotlib nltk
python -m spacy download en_core_web_sm
```
