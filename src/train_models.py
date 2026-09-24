import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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


def load_data():
    df = pd.read_csv(DATASET_PATH)

    # Remove the one row with a missing target label
    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    return X, y


def train_and_evaluate():
    X, y = load_data()

    print("=" * 60)
    print("PHISHING WEBSITE DETECTION")
    print("=" * 60)

    print("\nDataset:")
    print("Samples:", len(X))
    print("Features:", len(FEATURES))

    print("\nClass distribution:")
    print(y.value_counts())

    # Stratified split preserves the class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf"
        ),
    }

    results = []

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining: {name}")

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

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
            }
        )
    )


if __name__ == "__main__":
    train_and_evaluate()