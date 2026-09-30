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
5. Perform the initial stratified random 80/20 train-test split.
6. Perform a separate domain-grouped evaluation using normalized hostnames.

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

The initial evaluation uses a stratified random 80/20 hold-out split. A second
domain-grouped evaluation was added to assess generalization to previously
unseen hostnames. Both evaluations remove the single unlabeled record and
duplicate URLs and use the same deterministic 16 URL features and XGBoost
configuration.

### 1. Random stratified 80/20 evaluation

- Original records: 101,219
- Records with labels: 101,218
- Unique labeled URLs: 100,872
- URL features: 16
- Training samples: 80,697
- Test samples: 20,175

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 99.94% |
| Recall | 99.99% |
| F1 Score | 99.97% |

#### Random-split confusion matrix

```text
                 Predicted
              Phishing  Legitimate

Phishing        12698        1
Legitimate         7      7469
```

### 2. Domain-grouped evaluation

This evaluation uses normalized hostnames as groups with
`GroupShuffleSplit(test_size=0.20, random_state=42)`. No hostname appears in
both partitions.

| Metric | Score |
|---|---:|
| Training samples | 81,510 |
| Test samples | 19,362 |
| Unique train domains | 50,575 |
| Unique test domains | 12,644 |
| Domain overlap | 0 |
| Accuracy | 99.93285818% |
| Precision | 99.89933730% |
| Recall | 99.99160369% |
| F1 Score | 99.94544920% |

#### Domain-grouped confusion matrix

```text
[[11909, 1],
 [12, 7440]]
```

The existing random-split results remain available for direct comparison. The
domain-grouped evaluation is more informative for unseen-domain
generalization, while both results are dataset-specific and should not be
described as production performance.

## Leakage and Generalization Audit

- Exact raw-URL overlap after preprocessing: **0**
- Feature-level label leakage identified in the inspected 16 features: **none**
- Random-split test URLs whose hostname appeared in training:
  **8,859 / 20,175 = 43.9108%**
- Random-split test URLs using a training-seen hostname with a new path/query:
  **8,017 / 20,175 = 39.7373%**
- Domain-grouped train/test hostname overlap: **0**
- TLD popularity uses a fixed static set of 20 TLDs and is not label-derived.
- No timestamp column is available, so temporal validation was not performed.

These checks do not establish that the model is leak-free in an absolute
sense. The random split allows hostname reuse, which can make evaluation more
optimistic for unseen-domain deployment. The measured results should therefore
be interpreted as dataset-specific, with the domain-grouped evaluation
providing the more informative estimate for previously unseen hostnames.