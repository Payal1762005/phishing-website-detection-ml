from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import train_test_split

from feature_extraction import extract_features, FEATURE_NAMES


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "url_features_extracted1.csv"
)

MODEL_PATH = (
        PROJECT_ROOT
        / "models"
        / "xgboost_url_pipeline.pkl"
)

RESULTS_DIR = PROJECT_ROOT / "results"


# =========================================================
# Main
# =========================================================

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

    print(
        "After removing missing labels:",
        len(df)
    )

    # Remove duplicate URLs
    df = df.drop_duplicates(
        subset=["URL"]
    ).copy()

    print(
        "After removing duplicate URLs:",
        len(df)
    )

    # -----------------------------------------------------
    # 2. Extract features from raw URLs
    # -----------------------------------------------------

    print("\nExtracting features from raw URLs...")
    print("This may take a little while.")

    feature_rows = []

    total = len(df)

    for index, url in enumerate(
            df["URL"],
            start=1
    ):

        feature_rows.append(
            extract_features(str(url))
        )

        if index % 10000 == 0 or index == total:
            print(
                f"Processed {index:,} / "
                f"{total:,} URLs"
            )

    X = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES
    )

    y = (
        df["ClassLabel"]
        .astype(int)
        .reset_index(drop=True)
    )

    print(
        "\nFeature matrix shape:",
        X.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    print("\nClass distribution:")
    print(y.value_counts())

    # -----------------------------------------------------
    # 3. Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # -----------------------------------------------------
    # 4. XGBoost model
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

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # 5. Predictions
    # -----------------------------------------------------

    predictions = model.predict(X_test)

    # -----------------------------------------------------
    # 6. Metrics
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

    print("\n" + "=" * 70)
    print("END-TO-END XGBOOST RESULTS")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # -----------------------------------------------------
    # 7. Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    # -----------------------------------------------------
    # 8. Classification report
    # -----------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Phishing",
            "Legitimate"
        ]
    )

    print("\nClassification Report:")
    print(report)

    # -----------------------------------------------------
    # 9. Create results folder
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 10. Save metrics
    # -----------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "model": "XGBoost",
            "samples": len(df),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(FEATURE_NAMES),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    ])

    metrics_path = (
            RESULTS_DIR / "metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # -----------------------------------------------------
    # 11. Save classification report
    # -----------------------------------------------------

    report_path = (
            RESULTS_DIR
            / "classification_report.txt"
    )

    with open(
            report_path,
            "w",
            encoding="utf-8"
    ) as file:

        file.write(
            "Phishing Website Detection - "
            "XGBoost Classification Report\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(
            f"Samples: {len(df)}\n"
        )

        file.write(
            f"Training samples: {len(X_train)}\n"
        )

        file.write(
            f"Testing samples: {len(X_test)}\n"
        )

        file.write(
            f"Features: {len(FEATURE_NAMES)}\n\n"
        )

        file.write(report)

    # -----------------------------------------------------
    # 12. Save confusion matrix image
    # -----------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Phishing",
            "Legitimate"
        ]
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "XGBoost Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = (
            RESULTS_DIR
            / "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=200
    )

    plt.close()

    # -----------------------------------------------------
    # 13. Save feature importance
    # -----------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    importance_path = (
            RESULTS_DIR
            / "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # -----------------------------------------------------
    # 14. Save model
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    # -----------------------------------------------------
    # 15. Final output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nMetrics:")
    print(metrics_path)

    print("\nClassification report:")
    print(report_path)

    print("\nConfusion matrix:")
    print(confusion_path)

    print("\nFeature importance:")
    print(importance_path)

    print(
        "\nTraining pipeline completed successfully."
    )


if __name__ == "__main__":
    main()