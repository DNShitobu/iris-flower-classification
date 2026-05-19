# 🌸 Iris Flower Classification

A beginner-friendly ML project classifying iris flowers (Setosa, Versicolor, Virginica) using multiple scikit-learn classifiers, with full EDA and visualizations.

## 📌 Overview
| Detail | Value |
|--------|-------|
| Type | Multi-class Classification |
| Dataset | Iris (sklearn built-in, 150 samples) |
| Framework | scikit-learn |
| Models | Logistic Regression, Decision Tree, Random Forest, SVM, KNN |

## 🚀 Getting Started
```bash
git clone https://github.com/Dnshitobu/iris-flower-classification.git
cd iris-flower-classification
pip install -r requirements.txt
python iris_classifier.py
```

## 📊 What It Does
1. Loads Iris dataset and performs EDA (distributions, correlation, pairplot)
2. Trains 5 classifiers with StandardScaler preprocessing
3. Evaluates with accuracy, 5-fold CV, confusion matrix, classification report
4. Plots model comparison and Random Forest feature importances

## 📈 Sample Results
| Model | Test Acc | CV Acc |
|-------|:---:|:---:|
| Logistic Regression | ~0.967 | ~0.958 |
| **Random Forest** | **~1.000** | **~0.967** |
| SVM (RBF) | ~0.967 | ~0.967 |
| KNN | ~1.000 | ~0.967 |

## 🧠 Concepts Covered
Supervised classification · Feature scaling · Cross-validation · Confusion matrix · Feature importance
