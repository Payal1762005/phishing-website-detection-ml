# Phishing Website Detection ML: Leakage and Evaluation Audit

This is a temporary audit report. It does not change the training pipeline.

## Scope and method

The audit used `data/raw/url_features_extracted1.csv` and reproduced the
end-to-end preprocessing and split in `src/train_end_to_end.py`:

1. Read the CSV.
2. Drop the one row with a missing `ClassLabel`.
3. Drop duplicate raw `URL` values, keeping the first occurrence.
4. Split the remaining rows with `test_size=0.20`, `random_state=42`, and
   `stratify=ClassLabel`.

Hostnames were parsed with `urllib.parse.urlsplit`, lowercased, stripped of a
trailing dot, and IDNA-normalized where possible. No `www` removal was applied,
because that would merge distinct hostnames without an explicit project rule.

## A. What is already clean

- The end-to-end pipeline removes exact duplicate raw URLs before the split.
  Therefore, no exact duplicate raw URL can occur in both train and test in
  that pipeline: **0 cross-split exact duplicates**.
- Feature extraction in `src/feature_extraction.py` is row-local and operates
  only on the URL string. None of the 16 implementations reads `ClassLabel`,
  another row, the train/test partition, or a fitted lookup learned from the
  dataset.
- `tld_popularity` is not learned from this dataset. It is `int(tld.lower() in
  POPULAR_TLDS)` using a hard-coded set of 20 TLD strings in
  `src/tld_popularity.py`.
- The primary end-to-end evaluation extracts features before the split, but
  this is safe for these deterministic URL-only transformations; there is no
  fitted or label-dependent feature preprocessing.

## B. What could cause leakage or dataset bias

- The random row-level split permits the same hostname to appear in both
  partitions. This is not exact URL leakage, but it is a realistic source of
  optimistic performance when URL patterns are host-specific.
- **43.91% of test URLs** use a hostname also present in training. Of those,
  **8,017 test URLs** have a path/query resource not seen for that host in
  training; this is a near-duplicate/domain-reuse issue rather than exact URL
  duplication.
- The dataset has repeated raw URLs before preprocessing. The source CSV
  contains 101,219 rows, 648 rows belonging to repeated-URL groups, and 346
  duplicate occurrences beyond the first. The end-to-end pipeline removes
  these before splitting. Repeated URL groups do not have conflicting labels
  in this CSV.
- The CSV contains both raw `URL` and precomputed feature columns. The
  end-to-end script recomputes features from `URL`, while several legacy
  scripts consume the precomputed columns directly. The provenance of those
  stored feature columns is not established by the repository; this is a
  reproducibility risk, although the 99.96% end-to-end path does not use them.
- The dataset contains no collection timestamp/date column, so temporal
  ordering and temporal drift cannot be evaluated from this file.

## C. Exact counts and percentages

After dropping the one unlabeled row and duplicate raw URLs:

| Quantity | Result |
|---|---:|
| Rows after preprocessing | 100,872 |
| Train rows | 80,697 |
| Test rows | 20,175 |
| Raw duplicate rows in source CSV | 648 |
| Duplicate occurrences beyond first in source CSV | 346 |
| Exact raw URL duplicates across train/test | **0** |
| Unique normalized hostnames in train | **52,573** |
| Unique normalized hostnames in test | **15,295** |
| Hostnames appearing in both partitions | **4,650** |
| Test URLs on a training-seen hostname | **8,859 / 20,175 = 43.9108%** |
| Test URLs on a training-seen hostname with a new path/query | **8,017 / 20,175 = 39.7373%** |
| Test URLs with same hostname and same path/query as training | **842 / 20,175 = 4.1735%** |

The 842 same-host/same-path-query cases can still differ in other URL
components such as scheme, fragment, or spelling/casing. They are not counted
as exact raw URL duplicates; exact raw URL duplicates are zero after the
pipeline's deduplication step.

### Feature leakage review

All 16 features were inspected in `src/feature_extraction.py`:

| Feature | Uses labels/future data/full-dataset lookup? |
|---|---|
| `url_length` | No; URL-local |
| `has_ip_address` | No; URL-local |
| `dot_count` | No; URL-local |
| `https_flag` | No; URL-local |
| `url_entropy` | No; URL-local |
| `token_count` | No; URL-local |
| `subdomain_count` | No; URL-local |
| `query_param_count` | No; URL-local |
| `tld_length` | No; URL-local |
| `path_length` | No; URL-local |
| `has_hyphen_in_domain` | No; URL-local |
| `number_of_digits` | No; URL-local |
| `tld_popularity` | No; fixed static mapping |
| `suspicious_file_extension` | No; fixed code list and URL-local |
| `domain_name_length` | No; URL-local |
| `percentage_numeric_chars` | No; URL-local |

No feature uses `ClassLabel`, future/test information, labels aggregated from
the full dataset, or a lookup table built from complete-dataset records.

### TLD popularity

`src/tld_popularity.py` defines a fixed set:
`com, org, net, edu, gov, mil, int, co, io, ai, me, info, biz, in, uk, de,
fr, au, ca, us`.

It is therefore:

- a fixed external/static mapping: **Yes**
- calculated from the dataset: **No**
- calculated using labels: **No**
- calculated using all records before the split: **No**

## D. Interpretation of the current 99.96% result

The result should be described as **preliminary**, not fully validated.

There is no evidence in the inspected 16 feature implementations of direct
label or test-set leakage, and exact raw URL duplication is removed before the
split. However, the row-random split has substantial hostname overlap:
43.9108% of test URLs are on hosts represented in training, and 39.7373% are
new path/query variants on those hosts. Consequently, the reported score may
measure interpolation over reused domains more than performance on unseen
domains.

## E. Recommended additional experiment

Run a domain-grouped evaluation using `GroupShuffleSplit` with the normalized
hostname as `groups`, keeping all URLs from a hostname in only one partition.
Report accuracy, precision, recall, F1, confusion matrix, class counts, the
number of train/test domains, and the resulting test-domain coverage. Repeat
with several random seeds or grouped cross-validation if feasible.

Because no collection timestamp/date is present in the dataset columns, a
time-based hold-out evaluation is **not currently possible from this dataset**.

## Domain-grouped evaluation

Domain grouping was added as a separate evaluation rather than replacing the
original random stratified split. The original random split allows hostname
reuse: 4,650 hostnames appear in both partitions, covering 8,859 test URLs
(43.9108%). The grouped evaluation uses normalized hostnames with
`GroupShuffleSplit(test_size=0.20, random_state=42)`, so every hostname is
assigned to exactly one partition. It uses the same deterministic 16 URL
features and the same XGBoost configuration as the end-to-end evaluation.

The measured grouped results are:

| Metric | Result |
|---|---:|
| Train samples | 81,510 |
| Test samples | 19,362 |
| Unique train domains | 50,575 |
| Unique test domains | 12,644 |
| Domain overlap | **0** |
| Accuracy | 0.999329 |
| Precision | 0.998993 |
| Recall | 0.999916 |
| F1 | 0.999454 |

Grouped confusion matrix, with rows as actual `[Phishing, Legitimate]` and
columns as predicted `[Phishing, Legitimate]`:

```text
[[11909     1]
 [   12  7440]]
```

The separate outputs are saved as
`results/domain_grouped_metrics.csv`,
`results/domain_grouped_classification_report.txt`, and
`results/domain_grouped_confusion_matrix.png`. Existing random-split results
and the trained model artifact were not overwritten.
