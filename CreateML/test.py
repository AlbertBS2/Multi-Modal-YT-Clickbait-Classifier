import coremltools as ct
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix


def evaluate_model(model_path, csv_path):
    # 1. Load the model
    print("Loading model...")
    model = ct.models.MLModel(model_path)

    # Identify which features the model expects
    expected_features = [f.name for f in model.get_spec().description.input]

    # 2. Load the test data
    print("Loading data...")
    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        print("Error: Ground truth column 'label' not found in CSV. Use the exact name (case sensitive).")
        return

    y_true = []
    y_pred = []

    print(f"Running evaluation on {len(df)} rows...")

    for index, row in df.iterrows():
        # Only extract the features the model needs
        input_data = {feat: row[feat] for feat in expected_features if feat in row}

        try:
            prediction = model.predict(input_data)

            # Record the model's guess and the actual answer
            y_pred.append(int(prediction['label']))
            y_true.append(int(row['label']))
        except Exception as e:
            # Skip rows that fail (e.g. missing data)
            continue

    # 3. Calculate and Print Stats
    print("\n" + "=" * 30)
    print("   MODEL PERFORMANCE STATS")
    print("=" * 30)

    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(f"F1 Score:         {f1_score(y_true, y_pred):.4f}")
    print(f"Precision:        {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:           {recall_score(y_true, y_pred):.4f}")

    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    MODEL_PATH = "/Users/nainajnaho/PycharmProjects/yt-clickbait-comment-analysis/CreateML/MyTabularClassifier 1.mlmodel"
    CSV_PATH = "/Users/nainajnaho/PycharmProjects/yt-clickbait-comment-analysis/CreateML/datasets/test.csv"

    evaluate_model(MODEL_PATH, CSV_PATH)