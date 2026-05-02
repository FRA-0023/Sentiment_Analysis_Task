# Sentiment Analysis Task - JEMIB Project

A comprehensive sentiment analysis project implementing multiple machine learning models to classify text sentiment. This project is part of the JEMIB (Joint European Master's in Informatics: Bioinformatics) program at the University of Milan (UNIMIB).

## 📋 Project Overview

This project evaluates and compares various machine learning and deep learning approaches for sentiment analysis tasks. The models implemented range from classical algorithms (Logistic Regression, Naive Bayes, SVM) to state-of-the-art transformer-based models (BERT, DistilBERT, RoBERTa, Multilingual BERT).

## 📁 Project Structure

```
Sentiment_Analysis_Task/
├── Model/                              # Main model implementations
│   ├── main_model.ipynb               # Primary notebook for model execution
│   ├── common_utils.py                # Shared utility functions
│   ├── model_bert.py                  # BERT model implementation
│   ├── model_distilbert.py            # DistilBERT model implementation
│   ├── model_logistic_regression.py   # Logistic Regression model
│   ├── model_multilingual_bert.py     # Multilingual BERT model
│   ├── model_naive_bayes.py           # Naive Bayes model
│   ├── model_roberta.py               # RoBERTa model implementation
│   ├── model_svm.py                   # Support Vector Machine model
│   └── Result/                         # Model outputs and results
│
├── Data/                               # Dataset files
│   ├── SA_train_cleaned.csv           # Training data (cleaned)
│   ├── SA_test_cleaned.csv            # Test data (cleaned)
│   ├── SA_train_tokens.csv            # Training data (tokenized)
│   ├── SA_test_tokens.csv             # Test data (tokenized)
│   ├── sentiment_analysis_train.csv   # Original training dataset
│   └── sentiment_analysis_test.csv    # Original test dataset
│
├── Analysis/                           # Data analysis and reporting
│   ├── Data_Analysis_Phase1.tex       # Initial analysis report
│   ├── Data_Anlytics_Part1.Rmd        # Data analytics (R Markdown)
│   ├── Data_Anlytics_Full.Rmd         # Complete analytics report
│   └── Full_Project_Report.tex        # Final project report
│
└── README.md                           # This file
```

## 🧠 Implemented Models

The project includes implementations of the following models:

### Classical Machine Learning Models
- **Logistic Regression** (`model_logistic_regression.py`) - Baseline linear classifier
- **Naive Bayes** (`model_naive_bayes.py`) - Probabilistic classifier
- **Support Vector Machine** (`model_svm.py`) - Kernel-based classifier

### Deep Learning & Transformer Models
- **BERT** (`model_bert.py`) - Bidirectional Encoder Representations from Transformers
- **DistilBERT** (`model_distilbert.py`) - Distilled version of BERT (faster, lighter)
- **RoBERTa** (`model_roberta.py`) - Robustly Optimized BERT
- **Multilingual BERT** (`model_multilingual_bert.py`) - BERT trained on multiple languages

## 📊 Dataset

The project uses sentiment analysis datasets with the following structure:

- **Training Set**: `sentiment_analysis_train.csv` / `SA_train_cleaned.csv`
- **Test Set**: `sentiment_analysis_test.csv` / `SA_test_cleaned.csv`
- **Processed Versions**: Tokenized variants available (`*_tokens.csv`)

### Data Processing
- Raw data files are available for comparison
- Cleaned versions remove noise and preprocessing artifacts
- Tokenized versions provide pre-processed input for model training

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages (see below)

### Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install transformers torch scikit-learn pandas numpy
```

3. Navigate to the Model directory:
```bash
cd Model
```

### Running the Models

#### Option 1: Using the Main Notebook
```bash
jupyter notebook main_model.ipynb
```

#### Option 2: Running Individual Models
Each model can be run independently:
```python
from model_bert import BertModel
from model_logistic_regression import LogisticRegressionModel

# Initialize and train
model = BertModel()
model.train(training_data)
results = model.evaluate(test_data)
```

## 📈 Results

Model outputs, predictions, and performance metrics are stored in the `Model/Result/` directory.

## 📄 Reports and Analysis

- **Data Analysis Phase 1** (`Data_Analysis_Phase1.tex`) - Initial exploratory data analysis
- **Data Analytics Report** (`Data_Anlytics_Full.Rmd`) - Comprehensive statistical analysis
- **Full Project Report** (`Full_Project_Report.tex`) - Complete findings and conclusions

## 🛠️ Utilities

Common functions and shared utilities are provided in `common_utils.py`, including:
- Data loading and preprocessing
- Model evaluation metrics
- Visualization functions
- Helper functions for tokenization and encoding

## 📝 Notes

- The Multilingual BERT model is optimized for multilingual text classification
- DistilBERT offers a good balance between performance and computational efficiency
- Classical models (Logistic Regression, Naive Bayes, SVM) serve as baselines for comparison
- Results are reproducible with fixed random seeds

## 👥 Authors

JEMIB Project Team: Colombini Francesco & Hosen Jabed - University of Milan (UNIMIB)

## 📚 References

- BERT: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- DistilBERT: [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
- RoBERTa: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)

---

For questions or issues, please refer to the project documentation or contact the development team.
