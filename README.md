# 🧠 Sentiment Analysis Task: Classical ML vs. Deep Transformer Architectures

[![Language](https://img.shields.io/badge/Language-Python%203.10+%20%7C%20PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-BERT%20%7C%20RoBERTa%20%7C%20DistilBERT-blue?logo=huggingface)](https://huggingface.co/)
[![Report](https://img.shields.io/badge/Report-Complete%20Research%20Report-red?logo=adobeacrobatreader)](Full_Project_Report.pdf)

> Systematic empirical benchmark comparing classical linear algorithms (SVM, Naive Bayes, Logistic Regression) against fine-tuned Transformer models (BERT, RoBERTa, DistilBERT) on real-world sentiment classification.

---

## 📌 Executive Summary

Engineering teams frequently default to heavy, expensive Transformer models under the assumption that deep learning always justifies its inference cost. In production environments, latency, cloud inference budgets, and memory footprints must be traded off against marginal accuracy gains.

This project delivers an **exhaustive comparative benchmark** between classical statistical models and state-of-the-art pretrained Transformers:
- Evaluates performance across precision, recall, macro F1, and ROC-AUC metrics.
- Computes per-class confusion matrices to analyze misclassification patterns.
- Delivers a full quantitative analysis in [`Full_Project_Report.pdf`](Full_Project_Report.pdf).

---

## 🔍 Model Architecture Spectrum

| Architecture | Model Family | Key Structural Advantage | Inference Footprint |
|---|---|---|:---:|
| **Logistic Regression** | Linear / Generalized | Ultra-fast baseline; high interpretability | Minimal |
| **Multinomial Naive Bayes** | Probabilistic | Rapid text classification on sparse matrices | Minimal |
| **Linear / RBF SVM** | Margin Maximization | Robust margin boundaries in high-dimensional text | Low |
| **DistilBERT** | Distilled Transformer | 40% smaller and 60% faster than BERT; 97% language retention | Medium |
| **BERT (base-uncased)** | Bidirectional Transformer | Deep contextual language representation | High |
| **RoBERTa** | Optimized Transformer | Dynamically masked language modeling; superior semantic capture | High |
| **Multilingual BERT** | Multilingual Transformer | Cross-lingual zero-shot sentiment generalization | High |

---

## 🛠️ Repository Organization

```
Sentiment_Analysis_Task/
├── Model/
│   ├── main_model.ipynb               # Central execution pipeline
│   ├── common_utils.py                # Preprocessing, evaluation metrics & plotting
│   ├── model_logistic_regression.py   # Baseline linear model
│   ├── model_naive_bayes.py           # Probabilistic text classifier
│   ├── model_svm.py                   # Support Vector Machine implementation
│   ├── model_distilbert.py            # DistilBERT fine-tuning pipeline
│   ├── model_bert.py                  # Full BERT fine-tuning pipeline
│   ├── model_roberta.py               # RoBERTa optimization pipeline
│   └── Result/                        # Generated ROC curves and confusion matrices
├── Full_Project_Report.pdf            # Comprehensive 20+ page academic report
└── Data_Analysis_Phase1.pdf           # Initial exploratory data analysis report
```

---

**Author:** Francesco Colombini  
[GitHub Profile](https://github.com/FRA-0023) · [LinkedIn](https://www.linkedin.com/in/francescocolombini/)