from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from feature_extraction import extract_features, FEATURE_NAMES


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "url_features_extracted1.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_url_pipeline.pkl"


# ---------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("END-TO-END PHISHING URL DETECTION")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Original rows:", len(df))

    # Remove missing labels
    df = df.dropna(subset=["ClassLabel"]).copy()

    print("After removing missing labels:", len(df))

    # Remove duplicate URLs
    df = df.drop_duplicates(subset=["URL"]).copy()

    print("After removing duplicate URLs:", len(df))

    # -----------------------------------------------------
    # 2. Extract features from raw URLs
    # -----------------------------------------------------

    print("\nExtracting features from raw URLs...")
    print("This may take a little while.")

    feature_rows = []

    total = len(df)

    for index, url in enumerate(df["URL"], start=1):

        feature_rows.append(
            extract_features(str(url))
        )

        if index % 10000 == 0 or index == total:
            print(
                f"Processed {index:,} / {total:,} URLs"
            )

    X = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES
    )

    y = df["ClassLabel"].astype(int).reset_index(drop=True)

    # -----------------------------------------------------
    # 3. Show dataset information
    # -----------------------------------------------------

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nClass distribution:")
    print(y.value_counts())

    # -----------------------------------------------------
    # 4. Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # -----------------------------------------------------
    # 5. Train XGBoost
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 6. Predictions
    # -----------------------------------------------------

    predictions = model.predict(X_test)

    # -----------------------------------------------------
    # 7. Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    # -----------------------------------------------------
    # 8. Display results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("END-TO-END XGBOOST RESULTS")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
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

    # -----------------------------------------------------
    # 9. Save model
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()