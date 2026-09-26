# Phishing Website Detection using Machine Learning

A machine learning project for detecting potentially phishing URLs using URL-based feature extraction and XGBoost.

## Project Overview

Phishing websites are designed to imitate legitimate websites and trick users into revealing sensitive information. This project explores whether URL characteristics can be used to classify URLs as phishing or legitimate.

The project implements an end-to-end pipeline:

Raw URL → Feature Extraction → XGBoost Classification → Phishing / Legitimate Prediction

## Problem Statement

Traditional users often rely on visual appearance or website reputation to identify suspicious websites. This project investigates a machine learning approach that analyzes URL-level characteristics to identify phishing URLs.

## Approach

The project uses 16 URL-based characteristics:

- URL length
- IP address usage
- Number of dots
- HTTPS usage
- URL entropy
- Token count
- Subdomain count
- Query parameter count
- TLD length
- Path length
- Hyphens in the domain
- Number of digits
- TLD popularity
- Suspicious file extension
- Domain name length
- Percentage of numeric characters

These features are extracted from the raw URL and passed to an XGBoost classifier.

## Dataset

The project uses the LegitPhish dataset containing labeled phishing and legitimate URLs.

After preprocessing:

- Original records: 101,219
- Records with labels: 101,218
- Unique labeled URLs: 100,872
- Phishing URLs: 63,492
- Legitimate URLs: 37,380

The dataset label mapping used by this project is:

- `0` = Phishing
- `1` = Legitimate

## Data Preprocessing

The following preprocessing steps are applied:

1. Load the CSV dataset.
2. Remove records with missing labels.
3. Remove duplicate URLs.
4. Extract URL features from the raw URL.
5. Perform an 80/20 stratified train-test split.

## Model

The main classifier is XGBoost.

Configuration:

- Number of estimators: 200
- Maximum depth: 6
- Learning rate: 0.1
- Subsample: 0.8
- Column sampling: 0.8
- Random state: 42

## Results

The final end-to-end pipeline was evaluated using a stratified 80/20 train-test split after removing the single unlabeled record and duplicate URLs.

### Dataset used for evaluation

- Original records: 101,219
- Records with labels: 101,218
- Unique labeled URLs: 100,872
- Training samples: 80,697
- Test samples: 20,175
- URL features: 16

### XGBoost Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 99.94% |
| Recall | 99.99% |
| F1 Score | 99.97% |

### Confusion Matrix

```text
                 Predicted
              Phishing  Legitimate

Phishing        12698        1
Legitimate         7      7469