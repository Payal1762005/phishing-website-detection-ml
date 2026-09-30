from pathlib import Path
from urllib.parse import urlsplit

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from feature_extraction import FEATURE_NAMES, extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "url_features_extracted1.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def normalize_hostname(url):
    hostname = urlsplit(str(url)).hostname
    if not hostname:
        return ""

    hostname = hostname.rstrip(".").lower()
    try:
        return hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return hostname


def build_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )


def main():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["ClassLabel"]).copy()
    df = df.drop_duplicates(subset=["URL"]).copy()

    feature_rows = [
        extract_features(str(url))
        for url in df["URL"]
    ]
    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    y = df["ClassLabel"].astype(int).reset_index(drop=True)
    groups = df["URL"].map(normalize_hostname).reset_index(drop=True)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    train_domains = set(groups.iloc[train_indices])
    test_domains = set(groups.iloc[test_indices])
    overlap_count = len(train_domains & test_domains)

    if overlap_count != 0:
        raise RuntimeError(
            "Grouped split produced overlapping hostnames: "
            f"{overlap_count}"
        )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    confusion = confusion_matrix(y_test, predictions)
    metrics = {
        "evaluation": "domain_grouped",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "unique_train_domains": len(train_domains),
        "unique_test_domains": len(test_domains),
        "domain_overlap_count": overlap_count,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=0),
        "recall": recall_score(y_test, predictions, pos_label=0),
        "f1_score": f1_score(y_test, predictions, pos_label=0),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "domain_grouped_metrics.csv",
        index=False,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=["Phishing", "Legitimate"],
    )
    with open(
        RESULTS_DIR / "domain_grouped_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Domain-grouped XGBoost evaluation\n")
        file.write("=" * 60 + "\n\n")
        for name, value in metrics.items():
            file.write(f"{name}: {value}\n")
        file.write("\nConfusion matrix:\n")
        file.write(f"{confusion}\n\n")
        file.write(report)

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Phishing", "Legitimate"],
    )
    display.plot(values_format="d")
    plt.title("Domain-grouped XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "domain_grouped_confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    print(pd.Series(metrics).to_string())
    print("\nConfusion matrix:")
    print(confusion)


if __name__ == "__main__":
    main()
