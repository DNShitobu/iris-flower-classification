"""
Iris Flower Classification
==========================
A beginner-friendly ML project that classifies iris flowers into three species
(Setosa, Versicolor, Virginica) using multiple scikit-learn classifiers.

Covers:
  - Exploratory Data Analysis (EDA)
  - Feature visualization
  - Training multiple models
  - Evaluation: accuracy, confusion matrix, classification report
  - Model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Data ──────────────────────────────────────────────────────────────

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
target_names = iris.target_names

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 60)
print(f"\nDataset shape : {X.shape}")
print(f"Classes       : {list(target_names)}")
print(f"Class counts  :\n{y.value_counts().rename(dict(enumerate(target_names)))}\n")
print(X.describe().round(2))

# ── 2. EDA ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Iris Dataset — Feature Distributions by Species", fontsize=14, y=1.01)

for ax, feature in zip(axes.flatten(), X.columns):
    for label, name in enumerate(target_names):
        ax.hist(X[feature][y == label], bins=15, alpha=0.7, label=name)
    ax.set_title(feature)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

df_plot = X.copy()
df_plot["species"] = [target_names[i] for i in y]
pair = sns.pairplot(df_plot, hue="species", diag_kind="kde", corner=True)
pair.fig.suptitle("Iris Pairplot", y=1.02)
pair.savefig("eda_pairplot.png", dpi=150)
plt.close()

# ── 3. Preprocessing ──────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train & Evaluate Multiple Models ───────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    results[name] = {"test_accuracy": acc, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "y_pred": y_pred}
    print(f"\n{name}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV Accuracy   : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

best_name = max(results, key=lambda k: results[k]["test_accuracy"])
best_pred = results[best_name]["y_pred"]
print(classification_report(y_test, best_pred, target_names=target_names))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="mediumseagreen")
plt.title("Random Forest — Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ All done!")
