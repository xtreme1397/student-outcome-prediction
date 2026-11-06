# Student Dropout & Academic Success – Multi-class Classification

## a. Problem statement

Higher-education institutions lose a significant share of students before graduation. The goal of this project is to **predict, at enrolment and after the first two semesters, whether a student will _Drop out_, remain _Enrolled_ beyond the normal duration, or _Graduate_**, so that at-risk students can be identified early and supported.

This is framed as a **3-class supervised classification problem**. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data, evaluated with six metrics, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Value |
|---|---|
| Name | *Predict Students' Dropout and Academic Success* |
| Source | UCI Machine Learning Repository, ID 697 — https://archive.ics.uci.edu/dataset/697 |
| Instances | **4,424** students (no missing values) |
| Features | **36** (all numeric / integer-coded) |
| Target | `Target` — `Dropout` (1,421 · 32.1%), `Enrolled` (794 · 17.9%), `Graduate` (2,209 · 49.9%) |

Data comes from a Portuguese higher-education institution and covers students enrolled in 17 undergraduate degrees. The features fall into four groups:

- **Demographic** – marital status, nationality, gender, age at enrollment, displaced, international.
- **Socio-economic** – mother's/father's qualification and occupation, scholarship holder, debtor, tuition fees up to date, educational special needs.
- **Academic path at enrolment** – application mode & order, course, day/evening attendance, previous qualification (and its grade), admission grade.
- **Academic performance (1st & 2nd semester)** – curricular units credited / enrolled / evaluated / approved / grade / without evaluations.
- **Macro-economic** – unemployment rate, inflation rate, GDP.

**Pre-processing** (inside each saved scikit-learn `Pipeline`, so the app applies exactly the same steps):

- Header cleaned (the raw file has a UTF-8 BOM and a stray tab in one column name; separator is `;`).
- 9 nominal integer-coded columns (marital status, application mode, course, previous qualification, nationality, parents' qualifications & occupations) → **One-Hot encoded** (rare categories with < 10 rows grouped as "infrequent", unseen categories ignored).
- Remaining 27 numeric columns → **StandardScaler**.
- **Stratified 80 / 20 train–test split** (seed 2610) → 3,539 train rows / 885 test rows. The 885-row test split is saved as `test_data.csv`.
- Hyper-parameters tuned with **5-fold stratified GridSearchCV on the training set only** (scoring = macro-F1).

## c. GitHub repository link

**Repository:** https://github.com/xtreme1397/student-outcome-prediction

**Live Streamlit app:** https://student-outcome-prediction.streamlit.app

```
student-outcome-prediction/
├── app.py                 # Streamlit app (loads saved models, no retraining)
├── train_models.py        # training + tuning + evaluation script
├── ml_utils.py            # shared column cleaning and metric helpers
├── requirements.txt
├── README.md
├── test_data.csv          # held-out 20% test split (36 features + Target)
├── data/data.csv          # raw UCI dataset
├── .streamlit/config.toml # app theme
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── label_encoder.joblib
    ├── metrics.csv        # comparison table below
    └── best_params.json   # tuned hyper-parameters
```

### Run locally

```bash
pip install -r requirements.txt
python train_models.py      # optional – re-creates model/ and test_data.csv
streamlit run app.py
```

## d. Models used

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | `C = 0.5` (L2, lbfgs, multinomial) |
| Decision Tree | `max_depth = 6`, `min_samples_leaf = 20` |
| KNN | `n_neighbors = 7`, `weights = uniform` |
| Naive Bayes (Gaussian) | `var_smoothing = 0.1` |
| Support Vector Machine | RBF kernel, `C = 3.0`, `gamma = 0.01`, sigmoid-calibrated for probabilities |

### Comparison table (held-out test set, n = 885)

Precision, Recall and F1 are **macro-averaged** over the three classes; AUC is **one-vs-rest macro** AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7842 | **0.9017** | 0.7428 | 0.6896 | 0.6979 | 0.6422 |
| Decision Tree | 0.7525 | 0.8597 | 0.6992 | 0.6519 | 0.6620 | 0.5882 |
| KNN | 0.7345 | 0.8239 | 0.6839 | 0.6288 | 0.6382 | 0.5570 |
| Naive Bayes | 0.7096 | 0.8221 | 0.6526 | 0.6337 | 0.6404 | 0.5186 |
| Support Vector Machine | **0.7921** | 0.8654 | **0.7553** | **0.7035** | **0.7160** | **0.6555** |

> **Note:** When `train_models.py` was re-run on BITS Virtual Lab (Linux), every metric was identical except the Decision Tree AUC, which came out as **0.8599** instead of 0.8597. Decision trees can break ties between equally good splits differently across platforms, which slightly changes the predicted probabilities. The table above (and the deployed app) uses the committed models in `model/`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: 2nd best on accuracy, F1 and MCC and the **best AUC (0.902)**, i.e. its predicted probabilities rank students best. Most of the signal (approved units and grades in semester 1 & 2, tuition fees status) is close to linearly separable once scaled and one-hot encoded. Recall on *Dropout* is high (0.82) but *Enrolled* recall is only 0.32. |
| Decision Tree | Mid-table (Acc 0.753, MCC 0.588). Unpruned trees overfit, so CV picked a shallow tree (depth 6, ≥20 samples/leaf). It is the most interpretable model — 2nd-semester approved units alone carries 71% of the feature importance, followed by tuition-fees status (8%) — but axis-aligned splits give coarser probabilities, hence a lower AUC (0.860). |
| KNN | Weaker (Acc 0.735, MCC 0.557). With 130 dimensions after one-hot encoding, distances become less informative, and the many binary dummy columns dilute the few strong numeric predictors. It is biased toward the majority *Graduate* class (recall 0.93) and has the lowest *Enrolled* recall (0.26). |
| Naive Bayes | Lowest accuracy and MCC (0.710 / 0.519). The independence assumption is clearly violated (1st- and 2nd-semester enrolled/approved/grade features are strongly correlated, so evidence is double-counted) and many features are not Gaussian. Heavy `var_smoothing = 0.1` was needed. A side-effect is it predicts *Enrolled* more often, giving it the 2nd-best *Enrolled* recall (0.37). |
| Support Vector Machine | **Best overall**: top accuracy (0.792), macro precision, recall, F1 (0.716) and MCC (0.656). The RBF kernel captures mild non-linear interactions on top of the linear signal. It also has the best *Enrolled* F1 (0.47). Its AUC (0.865) is below Logistic Regression because SVM scores have to be calibrated into probabilities. Slowest model to train. |
| **Overall, winner for your dataset?** | **Support Vector Machine (RBF)** — best on 5 of 6 metrics, including the class-balance-aware MCC and macro-F1. Logistic Regression is a close runner-up and the better choice if well-ranked probabilities (AUC) or simplicity matter. Every model struggles with the minority *Enrolled* class (recall 0.26–0.37): these students sit between dropouts and graduates, so class re-weighting or resampling would be the next improvement. |

## Streamlit app features

- **CSV upload** of test data (`;` or `,` separated); the bundled `test_data.csv` can also be used or downloaded from the sidebar.
- **Model selection dropdown** covering all five saved models.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC as metric cards.
- **Confusion matrix** heatmap and **classification report** table.
- **Compare all models** tab: a highlighted metrics table, a bar chart for any chosen metric, and the best model by MCC.
- Download of per-row predictions with class probabilities.
- Column validation with clear error messages if the uploaded CSV is missing features or has unknown labels.
