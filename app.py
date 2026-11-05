"""Streamlit front-end: evaluate the pre-trained classifiers on an uploaded test CSV.

No training happens here - models are loaded from model/*.joblib and only used for prediction.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_utils import MODEL_FILES, TARGET_COL, compute_metrics, read_any_csv

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return models, encoder


@st.cache_data
def evaluate(_model, model_name, X, y_true, labels, data_key):
    """Cached on (model_name, data_key) so switching models / tabs is instant."""
    y_pred = _model.predict(X)
    y_proba = _model.predict_proba(X)
    return y_pred, y_proba, compute_metrics(y_true, y_pred, y_proba, labels)


models, label_encoder = load_artifacts()
class_names = list(label_encoder.classes_)
labels = list(range(len(class_names)))
expected_features = list(next(iter(models.values())).feature_names_in_)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"],
                                help="Must contain all 36 feature columns and the 'Target' column.")
    use_sample = st.toggle("Use bundled test_data.csv", value=uploaded is None,
                           disabled=uploaded is not None)
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.divider()
    with open(SAMPLE_CSV, "rb") as fh:
        st.download_button("⬇️ Download sample test_data.csv", fh, file_name="test_data.csv")
    st.caption("Models were trained offline by `train_models.py`. "
               "This app only loads them and predicts — nothing is retrained.")

st.title("🎓 Student Dropout & Academic Success — Model Evaluator")
st.write("Predicts whether a higher-education student will **Drop out**, stay **Enrolled**, "
         "or **Graduate** (UCI dataset #697). Upload a labelled test CSV and compare five classifiers.")

# ---------------- Load data ----------------
if uploaded is not None:
    df = read_any_csv(uploaded)
    data_key = f"upload:{uploaded.name}:{uploaded.size}"
elif use_sample:
    df = read_any_csv(SAMPLE_CSV)
    data_key = "sample"
else:
    st.info("👈 Upload a CSV or enable the bundled test data to begin.")
    st.stop()

missing = [c for c in expected_features + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"The CSV is missing {len(missing)} required column(s): {', '.join(missing[:10])}"
             + (" …" if len(missing) > 10 else ""))
    st.stop()

unknown = sorted(set(df[TARGET_COL].astype(str)) - set(class_names))
if unknown:
    st.error(f"Unknown target label(s) {unknown}. Expected one of {class_names}.")
    st.stop()

X = df[expected_features]
y_true = label_encoder.transform(df[TARGET_COL].astype(str))

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", len(expected_features))
c3.metric("Classes", " / ".join(class_names))
with st.expander("Preview data & class balance"):
    left, right = st.columns([3, 1])
    left.dataframe(df.head(20), width="stretch")
    right.dataframe(df[TARGET_COL].value_counts().rename("count"))

tab_single, tab_compare = st.tabs([f"🔍 {model_name}", "📊 Compare all models"])

# ---------------- Single model ----------------
with tab_single:
    y_pred, y_proba, m = evaluate(models[model_name], model_name, X, y_true, labels, data_key)

    st.subheader("Evaluation metrics")
    st.caption("Precision / Recall / F1 are macro-averaged; AUC is one-vs-rest macro.")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    out = df.copy()
    out["Predicted"] = label_encoder.inverse_transform(y_pred)
    for i, cls in enumerate(class_names):
        out[f"P({cls})"] = y_proba[:, i].round(4)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv")

# ---------------- All models ----------------
with tab_compare:
    rows = []
    for name, mdl in models.items():
        _, _, mm = evaluate(mdl, name, X, y_true, labels, data_key)
        rows.append({"ML Model Name": name, **mm})
    table = pd.DataFrame(rows).set_index("ML Model Name")

    st.subheader("Metric comparison")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
                 width="stretch")

    metric_choice = st.radio("Plot metric", list(table.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    sorted_tbl = table[metric_choice].sort_values()
    bars = ax.barh(sorted_tbl.index, sorted_tbl.values, color="#4c78a8")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric_choice)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    best = table["MCC"].idxmax()
    st.success(f"🏆 Best model on this data by MCC: **{best}** (MCC = {table.loc[best, 'MCC']:.4f})")
