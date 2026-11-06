# Credit Card Fraud Detection using XGBoost

## Project Overview

This project focuses on identifying fraudulent credit card transactions using **XGBoost**, a powerful gradient boosting algorithm.

The objective is to build a binary classification model that distinguishes legitimate transactions from fraudulent ones while addressing the challenge of highly imbalanced data.

## Dataset Information

The project uses the Credit Card Fraud Detection dataset.

| Description | Value |
|---|---:|
| Total transactions | 284,807 |
| Total features | 30 |
| Target column | `Class` |
| Fraudulent transactions | 492 |
| Legitimate transactions | 284,315 |

### Dataset Features

The dataset contains 30 input features:

- `Time` – Time elapsed between transactions.
- `V1` to `V28` – Anonymized features obtained through PCA transformation.
- `Amount` – Transaction amount.
- `Class` – Target variable.

**Target variable:**
- `0` – Legitimate transaction
- `1` – Fraudulent transaction

## Class Imbalance Analysis

One of the major challenges in this dataset is the significant difference between legitimate and fraudulent transactions.

The class distribution was examined using Pandas `value_counts()` and percentage distribution.

| Transaction Class | Count | Percentage |
|---|---:|---:|
| Legitimate (0) | 284,315 | 99.8273% |
| Fraudulent (1) | 492 | 0.1727% |
| **Total** | **284,807** | **100%** |

This shows that fraudulent transactions represent only approximately 0.17% of the entire dataset.

Because of this imbalance, accuracy alone is not a reliable measure of model performance.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Project Workflow

1. Dataset loading and inspection
2. Exploratory Data Analysis (EDA)
3. Class distribution and imbalance analysis
4. Feature and target separation
5. Training and testing data split
6. XGBoost model implementation
7. Handling class imbalance
8. Model evaluation
9. Prediction threshold analysis
10. Performance comparison

## Handling Class Imbalance

To address the highly imbalanced target variable, the XGBoost parameter `scale_pos_weight` was used.

The value used was:

`scale_pos_weight = 577.29`

This parameter increases the importance of the minority fraud class during model training.

## Model Evaluation

The model was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Precision-Recall AUC (PR-AUC)
- Confusion Matrix

### Baseline Model Results

| Metric | Score |
|---|---:|
| PR-AUC | 0.8534 |
| F1 Score | 0.883 |

### Tuned Model Results

At a prediction threshold of 0.5:

| Metric | Score |
|---|---:|
| Precision | 0.963 |
| Recall | 1.000 |
| F1 Score | 0.981 |

The tuned model achieved a higher F1-score than the baseline model.

## Key Learnings

- Understanding binary classification using XGBoost.
- Identifying and analyzing class imbalance.
- Working with highly imbalanced datasets.
- Using `scale_pos_weight` to address class imbalance.
- Understanding precision, recall, F1-score, and PR-AUC.
- Evaluating model performance beyond accuracy.
- Understanding prediction thresholds.

## Conclusion

This project provided practical experience in applying XGBoost to a highly imbalanced classification problem.

It highlighted the importance of class distribution analysis, appropriate evaluation metrics, and threshold selection when building fraud detection models.

## Author

**Nagaraj P**

AI & Machine Learning Student