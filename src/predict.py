import os
import sys
import joblib
import pandas as pd

# Allow imports from the src directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import extract_features, FEATURE_NAMES


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_url_pipeline.pkl"
)


def predict_url(url):
    # Extract features using the same pipeline used during training
    features = extract_features(url)

    # Keep the exact feature order used during training
    feature_values = [
        features[name]
        for name in FEATURE_NAMES
    ]

    X = pd.DataFrame(
        [feature_values],
        columns=FEATURE_NAMES
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    # Dataset labels:
    # 0 = Phishing
    # 1 = Legitimate
    if prediction == 0:
        label = "Phishing"
    else:
        label = "Legitimate"

    return label, probabilities


def main():
    print("=" * 60)
    print("PHISHING WEBSITE DETECTOR")
    print("=" * 60)

    url = input("\nEnter a URL: ").strip()

    if not url:
        print("Please enter a valid URL.")
        return

    try:
        prediction, probabilities = predict_url(url)

        print("\nPrediction:", prediction)

        print(
            f"Phishing probability: "
            f"{probabilities[0] * 100:.2f}%"
        )

        print(
            f"Legitimate probability: "
            f"{probabilities[1] * 100:.2f}%"
        )

    except FileNotFoundError:
        print("\nModel file not found.")
        print("Make sure you have run train_end_to_end.py first.")

    except Exception as error:
        print("\nPrediction failed.")
        print("Error:", error)


if __name__ == "__main__":
    main()