# Analyzing the Impact of Hate Speech Using NLP Techniques

> Predicting the social impact of online hate speech using Natural Language Processing, Transformer-based embeddings, and Machine Learning / Deep Learning models.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![NLP](https://img.shields.io/badge/NLP-HateSpeech-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Overview

Most existing hate speech detection systems focus only on determining whether a sentence contains hate speech.

This project goes one step further by predicting the **potential impact** of hate speech on individuals and society.

Instead of simply classifying text as hateful or non-hateful, the developed models estimate whether hateful content is likely to result in:

- Emotional Distress
- Provoking Violence
- Individual Harassment

The project compares multiple text embedding techniques and several machine learning and deep learning models to determine which combinations provide the most accurate impact prediction.

This work was completed as the **Minor Project** for the Bachelor of Technology (Computer Science Engineering) at **Jaypee Institute of Information Technology, Noida**.

---

# Objectives

The primary objectives of this project are:

- Analyze the impact of hate speech rather than only detecting it.
- Compare traditional and transformer-based text representations.
- Evaluate both Multilabel and Multiclass classification approaches.
- Compare different machine learning and deep learning architectures.
- Study the effectiveness of embedding fusion using transformer models.

---

# Dataset

Source:

- Hugging Face Hate Speech Dataset

Dataset Size:

- Approximately **135,556 samples**

The original dataset contains several ordinal annotations including:

- Sentiment
- Respect
- Insult
- Dehumanize
- Humiliate
- Status
- Violence
- Genocide
- Attack / Defend
- Hate Speech

These annotations were further engineered into three impact categories used throughout this project.

---

# Target Labels

The original annotations were transformed into three impact prediction labels:

| Label | Description |
|--------|-------------|
| Emotional Distress | Predicts emotional harm caused by hateful content |
| Provoking Violence | Predicts whether the content encourages violent behaviour |
| Individual Harassment | Predicts targeted harassment towards individuals |

The project explores both:

- **Multilabel Classification** (multiple labels may be true simultaneously)
- **Multiclass Classification** (single class prediction)

---

# Complete Project Workflow

```
                Raw Dataset
                     │
                     ▼
        Data Cleaning & Preprocessing
                     │
                     ▼
           Feature Engineering
                     │
                     ▼
      Creation of Impact Labels
                     │
                     ▼
        Text Representation Learning
     ┌────────────┬────────────┬──────────────┐
     │            │            │
     ▼            ▼            ▼
   TF-IDF      BERT/SBERT    GPT-Neo
                    │
                    ▼
      BERT + GPT-Neo / SBERT + GPT-Neo
             Embedding Fusion
                    │
                    ▼
        Machine Learning Models
     ┌────────────┬──────────────┬─────────────┐
     │            │              │
     ▼            ▼              ▼
   MLP         BiLSTM        XGBoost
                    │
                    ▼
     CNN-BiLSTM-Attention
                    │
                    ▼
         Model Evaluation
                    │
                    ▼
 Accuracy • Precision • Recall • F1-score
```

---

# Text Preprocessing

The preprocessing pipeline includes:

- Dataset loading
- Missing value handling
- Text cleaning
- Tokenization
- Stop-word removal
- Label engineering
- Binary label conversion
- Train-Test splitting

---

# Embedding Techniques

## TF-IDF

Traditional statistical text representation used as the baseline model.

---

## BERT / Sentence-BERT (SBERT)

Contextual transformer embeddings capable of capturing semantic relationships between words and sentences.

---

## GPT-Neo

Transformer-based embeddings extracted using GPT-Neo for richer contextual understanding.

---

## Embedding Fusion

Experiments were conducted by concatenating BERT/SBERT embeddings with GPT-Neo embeddings to evaluate whether combining multiple semantic representations improves classification performance.

---

# Machine Learning Models

## Multilabel Classification

- Multi Layer Perceptron (MLP)
- Hierarchical CNN-BiLSTM-Attention Network

---

## Multiclass Classification

- Bidirectional LSTM (BiLSTM)
- XGBoost

---

# Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Weighted F1 Score

The experiments compare different embeddings and architectures for both Multiclass and Multilabel impact prediction.

---

# Repository Files

Note: Each notebook represents a separate experiment evaluating different embedding techniques (TF-IDF, BERT, SBERT, GPT-Neo, and embedding fusion) and classification models. The notebooks are designed to compare methodologies rather than be executed strictly in sequence.

---

## 📓 01_tfidf_baseline(1).ipynb

Implements the baseline NLP pipeline using **TF-IDF features** instead of transformer embeddings.

Contents:

- Dataset preprocessing
- TF-IDF vectorization
- Baseline feature generation
- Model training
- Performance comparison with transformer embeddings

Purpose:

Establishes a traditional machine learning baseline for evaluating the effectiveness of contextual embeddings.

---

## 📓 02_gptneo_embeddings(1).ipynb

Generates contextual embeddings using **GPT-Neo**.

Contents:

- GPT-Neo tokenizer
- GPT-Neo embedding extraction
- Embedding generation pipeline
- Export of generated embeddings for downstream experiments

Purpose:

Creates transformer embeddings that are later used for multiple classification models.

---

## 📓 bert_multilabel_classification.ipynb

Implements the complete **Multilabel Classification** pipeline using **BERT embeddings**.

Contents:

- Dataset preprocessing
- Label engineering
- BERT embedding generation
- Label distribution analysis
- Multi Layer Perceptron
- CNN-BiLSTM-Attention Network
- Graph-based experimentation
- Performance evaluation
- Prediction examples

Purpose:

Predicts multiple hate speech impact labels simultaneously.

---

## 📓 bert_multiclass_classification.ipynb

Implements Multiclass impact prediction using **BERT embeddings**.

Contents:

- BERT embeddings
- BiLSTM
- XGBoost
- Model evaluation
- Classification reports

Purpose:

Evaluates traditional multiclass prediction using contextual embeddings.

---

## 📓 gptneo_multiclass_classification.ipynb

Implements Multiclass classification using **GPT-Neo embeddings**.

Contents:

- GPT-Neo embeddings
- BiLSTM
- XGBoost
- Classification reports
- Performance comparison

Purpose:

Studies GPT-Neo representations for multiclass hate speech impact prediction.

---

## 📓 sbert_multiclass_classification.ipynb

Uses **Sentence-BERT (SBERT)** embeddings for multiclass prediction.

Contents:

- SBERT embedding generation
- BiLSTM
- XGBoost
- MLP experiments
- Evaluation metrics

Purpose:

Compares Sentence-BERT with other transformer embedding techniques.

---

## 📓 sbert&gptneoembeddingsclassification.ipynb

Experiments with **Sentence-BERT and GPT-Neo** embeddings.

Contents:

- SBERT embeddings
- GPT-Neo embeddings
- Multilabel classification
- Prediction examples
- Performance evaluation

Purpose:

Investigates how different transformer embeddings influence multilabel prediction.

---

## 📓 bert_gptneo_multilabel_fusion.ipynb

Implements **Embedding Fusion** by combining BERT and GPT-Neo representations.

Contents:

- BERT embeddings
- GPT-Neo embeddings
- Embedding concatenation
- CNN-BiLSTM-Attention architecture
- Classification reports
- Label distribution analysis

Purpose:

Evaluates whether combining contextual embeddings improves multilabel classification performance.

---

## 📄 Minor Project Report.pdf

Complete academic report describing:

- Literature Survey
- Problem Statement
- Methodology
- Dataset
- Feature Engineering
- Model Architectures
- Experimental Results
- Conclusions
- Future Scope

---

# Technologies Used

Programming Language

- Python

Libraries

- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# Experimental Summary

The project compares:

### Embedding Techniques

- TF-IDF
- BERT
- Sentence-BERT
- GPT-Neo
- BERT + GPT-Neo Fusion
- SBERT + GPT-Neo

Across

### Machine Learning Models

- MLP
- BiLSTM
- XGBoost
- CNN-BiLSTM-Attention

For

- Multilabel Classification
- Multiclass Classification

This enables a comprehensive comparison between traditional statistical features, transformer-based contextual embeddings, and embedding fusion techniques.

---

# Future Work

Potential future extensions include:

- Multilingual hate speech impact prediction
- Explainable AI (SHAP / LIME)
- Real-time deployment using FastAPI or Streamlit
- Multi-modal hate speech analysis using text and images
- Larger transformer models such as RoBERTa, DeBERTa and LLaMA

---

# Authors

**Dev Agarwal**

B.Tech Computer Science Engineering

Jaypee Institute of Information Technology, Noida

Contributors

- Dev Agarwal
- Alokik Garg
- Swapnil Pandey

Supervisor

Dr. Sayani Ghosal

---

# Citation

If you use this work, please cite:

```bibtex
@misc{hate_speech_impact_analysis,
title={Analyzing the Impact of Hate Speech Using NLP Techniques},
author={Dev Agarwal and Alokik Garg and Swapnil Pandey},
year={2024},
note={B.Tech Minor Project, Jaypee Institute of Information Technology}
}
```

---

# License

This repository is intended for academic and research purposes.
