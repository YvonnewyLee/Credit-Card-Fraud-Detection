# Credit Card Fraud Detection Using Gaussian Mixture Model

## Overview
This project applies anomaly detection techniques using a Gaussian Mixture Model (GMM) to detect fraudulent transactions in a credit card fraud detection dataset. The dataset contains transactions made by European cardholders in September 2013 and is highly imbalanced, with fraud transactions making up only 0.17% of the total.

The project involves various steps, from data preprocessing to model evaluation, using GMM to identify fraudulent transactions based on multiple feature sets.

## Key Components
1. **Data Preprocessing**:
   - The dataset is loaded from Kaggle and consists of 30 features (V1-V28, Time, Amount), with Time and Amount as the only non-transformed features.
   - The dataset is imbalanced, with only 0.17% fraud transactions, requiring specific handling for model training and evaluation.

2. **Modeling**:
   - **Part 1**: The data is split into training, validation, and test sets. The distribution of fraud and non-fraud transactions is visualized across different features to identify the most informative ones.
   - **Part 2**: A Gaussian Mixture Model is fit on single features (V1-V28) to find the best features for fraud detection based on the AUC score.
   - **Part 3**: A multi-feature Gaussian Mixture Model is fit on pairs of features, and the optimal number of components is determined.
   - **Part 4**: A more complex model uses two Gaussian distributions: one for non-fraud and another for fraud transactions, finding an optimal threshold to maximize F1 score.

3. **Model Comparison**:
   - Various models are tested, comparing different feature sets and Gaussian distributions (single vs. multiple components).
   - The models are evaluated using AUC, F1 score, precision, and recall, with the best model being a 2-Gaussian mixture model with five features (V14, V10, V4, V16, V11), achieving an F1 score of 0.8095 on the validation set.

4. **Evaluation on Test Set**:
   - The best model is tested on the unseen test set, yielding an F1 score of **0.8293**, precision of **0.8095**, and recall of **0.85**.

## Key Steps
1. **Data Loading & Exploration**:
   - Dataset is loaded and split into training, validation, and test sets.
   - Feature distributions are analyzed to identify the most informative features for distinguishing fraudulent transactions.

2. **Gaussian Mixture Model**:
   - Fit a Gaussian distribution to single features, followed by multi-feature models using Gaussian Mixtures.
   - Compare different configurations of Gaussian distributions (1, 2, 3, etc.) for fraud and non-fraud transactions.

3. **Model Optimization**:
   - The best features and thresholds are selected based on AUC and F1 scores.
   - Optimal threshold selection maximizes F1 scores for detecting fraudulent transactions.

4. **Model Evaluation**:
   - Models are evaluated on training, validation, and test datasets.
   - The best model (5 features, 2 Gaussians, 1 for non-fraud and 3 for fraud) achieves the highest F1 score on the test set.

## Results Summary
- The best-performing model involves using **5 features** (V14, V10, V4, V16, V11) with a **2 Gaussian mixture model**:
  - **F1 Score (Test Set)**: 0.8293
  - **Precision**: 0.8095
  - **Recall**: 0.85
  - **Best Threshold (c)**: 2.8

## Dependencies
- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `sklearn`
  - `wget`

## Conclusion
This project demonstrates the use of Gaussian Mixture Models for credit card fraud detection, successfully identifying fraudulent transactions with a high F1 score. By leveraging unsupervised learning methods and optimizing feature selection and thresholding, the model performs well on imbalanced datasets.
