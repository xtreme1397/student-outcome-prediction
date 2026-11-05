"""Train the 5 classifiers on the UCI student dropout dataset and save them to model/.

Run:  python train_models.py
Outputs:
  model/*.joblib            one fitted sklearn Pipeline (preprocessing + classifier) per model
  model/label_encoder.joblib
  model/metrics.csv         comparison table on the held-out test split
  test_data.csv             the held-out 20% split (features + Target) for the Streamlit app
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import MODEL_FILES, NOMINAL_COLS, TARGET_COL, compute_metrics, read_any_csv

ROOT = Path(__file__).parent
RAW_DATA = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "model"
SEED = 2610


def build_preprocessor(numeric_cols):
    return ColumnTransformer(
        [
            ("nominal", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), NOMINAL_COLS),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )


def candidate_models():
    """Classifier + a small hyper-parameter grid for each (tuned with 5-fold CV on train only)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.05, 0.1, 0.5, 1.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [5, 20, 50]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [7, 15, 25, 35], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
        ),
        "Support Vector Machine": (
            # SVC has no native probabilities; calibration gives predict_proba for the AUC.
            CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED), ensemble=False),
            {"clf__estimator__C": [0.5, 1.0, 3.0], "clf__estimator__gamma": ["scale", 0.01]},
        ),
    }


def main():
    df = read_any_csv(RAW_DATA)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1] - 1} features")
    print(df[TARGET_COL].value_counts(), "\n")

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]
    numeric_cols = [c for c in X.columns if c not in NOMINAL_COLS]

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED
    )

    # Hold-out split is what gets uploaded to the Streamlit app.
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test_raw
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    le = LabelEncoder().fit(y_raw)
    y_train, y_test = le.transform(y_train_raw), le.transform(y_test_raw)
    labels = list(range(len(le.classes_)))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}
    for name, (clf, grid) in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric_cols)), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_proba, labels)
        rows.append({"ML Model Name": name, **m})
        best_params[name] = {k.replace("clf__", "").replace("estimator__", ""): v for k, v in search.best_params_.items()}

        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
        print(f"{name:<24} best={best_params[name]}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    try:
        print("\n" + metrics.to_markdown(index=False))
    except ImportError:  # to_markdown needs the optional 'tabulate' package
        print("\n" + metrics.to_string(index=False))


if __name__ == "__main__":
    main()
