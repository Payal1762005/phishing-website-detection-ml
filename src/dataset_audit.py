import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATASET_PATH = "../data/raw/url_features_extracted1.csv"

FEATURES = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "tld_popularity",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]

TARGET = "ClassLabel"


def main():
    df = pd.read_csv(DATASET_PATH)

    print("=" * 60)
    print("XGBOOST - DEDUPLICATED DATASET")
    print("=" * 60)

    print("\nOriginal rows:", len(df))

    # Remove rows without a target label
    df = df.dropna(subset=[TARGET]).copy()

    print("After removing missing labels:", len(df))

    # Keep only one record for each URL
    df = df.drop_duplicates(subset=["URL"]).copy()

    print("After removing duplicate URLs:", len(df))
    print("Unique URLs:", df["URL"].nunique())

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    print("\nTraining XGBoost...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=0
    )

    print("\n" + "=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Phishing",
                "Legitimate"
            ]
        )
    )


if __name__ == "__main__":
    main()