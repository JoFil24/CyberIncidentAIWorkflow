import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

DATA_FILE = os.path.join("Datasets", "part-00000-0af89d10-df53-44fd-b124-a8a496fd5023-c000(2).csv")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the SOC dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Preprocess dataset for training a Decision Tree classifier."""
    drop_columns = ["uid", "community_id", "datetime", "src_ip", "dest_ip", "ts"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

    target_column = "mitre_attack_tactics"
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    y = df[target_column].astype(str)
    X = df.drop(columns=[target_column])

    encoders: dict[str, LabelEncoder] = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            encoders[col] = encoder

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    encoders[target_column] = target_encoder

    print("Preprocessing complete:")
    print(f"  Features: {X.columns.tolist()}")
    print(f"  Target classes: {list(target_encoder.classes_)}")
    return X, y_encoded, encoders


def train_decision_tree(X: pd.DataFrame, y: np.ndarray) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier and return the fitted model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_leaf=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model evaluation")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("  Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    feature_importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda item: item[1], reverse=True
    )
    print("  Feature importances:")
    for feature, importance in feature_importances:
        print(f"    {feature}: {importance:.4f}")

    return model


def plot_tree_model(model: DecisionTreeClassifier, X: pd.DataFrame, output_path: str = "decision_tree_plot.png") -> None:
    """Plot and save the trained decision tree structure."""
    plt.figure(figsize=(18, 12))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=[str(c) for c in np.unique(model.classes_)],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved tree plot to {output_path}")


def main() -> None:
    csv_path = DATA_FILE
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = load_data(csv_path)
    X, y, _ = preprocess(df)
    model = train_decision_tree(X, y)
    try:
        plot_tree_model(model, X)
    except Exception as exc:
        print(f"Unable to plot tree: {exc}")


if __name__ == "__main__":
    main()
