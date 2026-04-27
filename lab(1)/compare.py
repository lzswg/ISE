# -*- coding: utf-8 -*-
"""
Bug report classification experiments.

This script combines the baseline model and improved models into one file.
It includes:
1) GaussianNB + TF-IDF baseline
2) Logistic Regression + TF-IDF
3) Linear SVM + Word/Char TF-IDF
4) Complement Naive Bayes + TF-IDF

Features:
- Repeated train/test splits
- Grid search tuning
- Detailed console output for every run
- CSV export for detailed and summary results
- Comparison plots
"""

########## 1. Import required libraries ##########

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, GaussianNB

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


########## 2. Global configuration ##########

RANDOM_STATE = 42
REPEAT = 10

project = "pytorch"
raw_datafile = f"{project}.csv"
processed_datafile = "Title+Body.csv"

detail_csv = f"./{project}_all_models_detailed_results.csv"
summary_csv = f"./{project}_all_models_summary_results.csv"

fig_bar = f"./{project}_model_comparison_bar.png"
fig_line = f"./{project}_repeat_f1_line.png"


########## 3. Text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r"<.*?>")
    return html.sub(" ", str(text))


def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(" ", str(text))


try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words_list = list(ENGLISH_STOP_WORDS)
except Exception:
    stop_words_list = []

custom_stop_words_list = ["..."]
final_stop_words_list = set(stop_words_list + custom_stop_words_list)


def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters
    and converting it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", str(string))
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def normalize_text(text):
    """Apply all cleaning steps together."""
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = clean_str(text)
    return text


########## 4. Data preparation ##########

def prepare_data(raw_path, processed_path):
    """
    Read data and prepare a unified CSV with text and sentiment columns.
    If processed_path already exists and is valid, use it directly.
    Otherwise, build it from the raw file.
    """
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path).fillna("")
        if {"text", "sentiment"}.issubset(df.columns):
            return df

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Data file not found: {raw_path} or {processed_path}")

    pd_all = pd.read_csv(raw_path).fillna("")
    pd_all = pd_all.sample(frac=1, random_state=999).reset_index(drop=True)

    if {"Title", "Body", "class"}.issubset(pd_all.columns):
        pd_all["Title+Body"] = pd_all.apply(
            lambda row: row["Title"] + ". " + row["Body"] if str(row["Body"]).strip() != "" else str(row["Title"]),
            axis=1
        )

        rename_map = {"class": "sentiment", "Title+Body": "text"}
        if "Unnamed: 0" in pd_all.columns:
            rename_map["Unnamed: 0"] = "id"

        pd_tplusb = pd_all.rename(columns=rename_map)
        keep_cols = [c for c in ["id", "Number", "sentiment", "text"] if c in pd_tplusb.columns]
        pd_tplusb.to_csv(processed_path, index=False, columns=keep_cols)
        return pd_tplusb[keep_cols].fillna("")

    if {"text", "sentiment"}.issubset(pd_all.columns):
        return pd_all[["text", "sentiment"]].copy().fillna("")

    raise ValueError("The data file does not contain usable columns. Expected text/sentiment or Title/Body/class.")


def load_and_clean_data():
    """Load data and clean the text column."""
    data = prepare_data(raw_datafile, processed_datafile)
    data = data.fillna("")
    data["text"] = data["text"].astype(str).apply(normalize_text)
    return data


########## 5. Metric computation ##########

def get_average_type(y_true):
    """Use binary averaging for binary labels, macro otherwise."""
    n_classes = len(np.unique(y_true))
    return "binary" if n_classes == 2 else "macro"


def get_auc_score(estimator, X_test, y_test, n_classes):
    """
    Compute AUC in a robust way for binary and multiclass settings.
    """
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)
    else:
        return np.nan

    try:
        if n_classes == 2:
            if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                y_score = y_score[:, 1]
            return roc_auc_score(y_test, y_score)
        else:
            return roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
    except Exception:
        return np.nan


def evaluate_model(estimator, X_test, y_test, n_classes):
    """Evaluate a trained model on the test set."""
    y_pred = estimator.predict(X_test)
    avg_type = get_average_type(y_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=avg_type, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg_type, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg_type, zero_division=0),
        "AUC": get_auc_score(estimator, X_test, y_test, n_classes),
    }
    return metrics


########## 6. Helper transformer for GaussianNB ##########

class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense matrices for GaussianNB."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)


########## 7. Model builders ##########

def build_gaussiannb_pipeline():
    """
    GaussianNB + TF-IDF baseline.
    """
    pipeline = Pipeline([
        ("vect", TfidfVectorizer(lowercase=False)),
        ("to_dense", DenseTransformer()),
        ("clf", GaussianNB())
    ])

    param_grid = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "vect__min_df": [1, 2],
        "vect__max_df": [0.90, 0.95],
        "vect__max_features": [1000, 2000],
        "vect__sublinear_tf": [True],
        "clf__var_smoothing": np.logspace(-12, 0, 13)
    }
    return pipeline, param_grid


def build_logreg_pipeline():
    """
    Logistic Regression with TF-IDF.
    """
    pipeline = Pipeline([
        ("vect", TfidfVectorizer(lowercase=False)),
        ("clf", LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])

    param_grid = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "vect__min_df": [1, 2],
        "vect__max_df": [0.90, 0.95],
        "vect__sublinear_tf": [True],
        "vect__strip_accents": [None, "unicode"],
        "clf__C": [0.25, 0.5, 1.0, 2.0],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
        "clf__class_weight": [None, "balanced"],
    }
    return pipeline, param_grid


def build_linearsvm_pipeline():
    """
    Linear SVM with combined word and character TF-IDF features.
    """
    word_vec = TfidfVectorizer(lowercase=False)
    char_vec = TfidfVectorizer(lowercase=False, analyzer="char_wb")

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", word_vec),
            ("char", char_vec),
        ])),
        ("clf", LinearSVC(random_state=RANDOM_STATE, dual=True))
    ])

    param_grid = {
        "features__word__ngram_range": [(1, 1), (1, 2)],
        "features__word__min_df": [1, 2],
        "features__word__max_df": [0.90, 0.95],
        "features__word__sublinear_tf": [True],
        "features__char__ngram_range": [(3, 5), (4, 6)],
        "features__char__min_df": [1, 2],
        "features__char__sublinear_tf": [True],
        "clf__C": [0.25, 0.5, 1.0, 2.0],
        "clf__class_weight": [None, "balanced"],
    }
    return pipeline, param_grid


def build_complementnb_pipeline():
    """
    Complement Naive Bayes with TF-IDF.
    """
    pipeline = Pipeline([
        ("vect", TfidfVectorizer(lowercase=False)),
        ("clf", ComplementNB())
    ])

    param_grid = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "vect__min_df": [1, 2, 5],
        "vect__max_df": [0.90, 0.95],
        "vect__sublinear_tf": [True],
        "vect__use_idf": [True],
        "vect__smooth_idf": [True],
        "clf__alpha": [0.1, 0.25, 0.5, 1.0],
        "clf__norm": [False, True],
    }
    return pipeline, param_grid


MODEL_BUILDERS = {
    "GaussianNB_TFIDF": build_gaussiannb_pipeline,
    "LogReg_TFIDF": build_logreg_pipeline,
    "LinearSVM_WordCharTFIDF": build_linearsvm_pipeline,
    "ComplementNB_TFIDF": build_complementnb_pipeline,
}


########## 8. Training and tuning ##########

def run_one_model(
    model_name,
    pipeline,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    n_classes,
):
    """
    Fit GridSearchCV, select the best model, and evaluate it on the test set.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    test_metrics = evaluate_model(best_model, X_test, y_test, n_classes)

    result = {
        "Model": model_name,
        "Best_CV_F1": grid.best_score_,
        "Best_Params": json.dumps(grid.best_params_, ensure_ascii=False),
        **test_metrics
    }
    return result


########## 9. Visualization ##########

def plot_comparison(summary_df, save_path=None):
    """Draw a comparison bar chart for Accuracy, F1, and AUC."""
    x = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(13, 6))
    plt.bar(x - width, summary_df["Accuracy_Mean"], width=width, label="Accuracy")
    plt.bar(x, summary_df["F1_Mean"], width=width, label="F1")
    plt.bar(x + width, summary_df["AUC_Mean"], width=width, label="AUC")

    plt.xticks(x, summary_df["Model"], rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Model Comparison on Mean Test Metrics")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_repeat_f1(detail_df, save_path=None):
    """Draw the F1 trend across repeats for each model."""
    plt.figure(figsize=(13, 6))
    for model_name in detail_df["Model"].unique():
        sub = detail_df[detail_df["Model"] == model_name].sort_values("Repeat")
        plt.plot(sub["Repeat"], sub["F1"], marker="o", label=model_name)

    plt.xlabel("Repeat")
    plt.ylabel("F1")
    plt.title("F1 Score Across Repeats")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


########## 10. Main function ##########

def main():
    # Load and preprocess data
    data = load_and_clean_data()

    # Encode labels
    le = LabelEncoder()
    data["sentiment_enc"] = le.fit_transform(data["sentiment"])

    X = data["text"].astype(str).values
    y = data["sentiment_enc"].values
    n_classes = len(np.unique(y))

    print("Label mapping:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i} -> {cls}")

    print(f"\nNumber of classes: {n_classes}")
    print(f"Number of samples:  {len(data)}\n")

    all_run_results = []

    for repeat_idx in range(REPEAT):
        print(f"========== Repeat {repeat_idx + 1}/{REPEAT} ==========")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE + repeat_idx,
            stratify=y
        )

        for model_name, builder in MODEL_BUILDERS.items():
            print(f"Training {model_name} ...")
            pipeline, param_grid = builder()

            result = run_one_model(
                model_name=model_name,
                pipeline=pipeline,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_classes=n_classes,
            )

            result["Repeat"] = repeat_idx + 1
            all_run_results.append(result)

            print(
                f"  Best_CV_F1={result['Best_CV_F1']:.4f} | "
                f"Acc={result['Accuracy']:.4f} | "
                f"Prec={result['Precision']:.4f} | "
                f"Recall={result['Recall']:.4f} | "
                f"F1={result['F1']:.4f} | "
                f"AUC={result['AUC']:.4f}"
            )
            print(f"  Best_Params={result['Best_Params']}\n")

        print()

    # Save detailed results
    detail_df = pd.DataFrame(all_run_results)
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    # Aggregate summary
    summary_df = (
        detail_df
        .groupby("Model")
        .agg(
            Repeat_Count=("Repeat", "count"),
            CV_F1_Mean=("Best_CV_F1", "mean"),
            CV_F1_Std=("Best_CV_F1", "std"),
            Accuracy_Mean=("Accuracy", "mean"),
            Accuracy_Std=("Accuracy", "std"),
            Precision_Mean=("Precision", "mean"),
            Precision_Std=("Precision", "std"),
            Recall_Mean=("Recall", "mean"),
            Recall_Std=("Recall", "std"),
            F1_Mean=("F1", "mean"),
            F1_Std=("F1", "std"),
            AUC_Mean=("AUC", "mean"),
            AUC_Std=("AUC", "std"),
        )
        .reset_index()
        .sort_values(by="F1_Mean", ascending=False)
    )

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Console output
    print("\n================ Detailed Results (Head) ================")
    print(detail_df.head().to_string(index=False))

    print("\n================ Overall Summary ================")
    print(summary_df.to_string(index=False))

    best_model_row = summary_df.iloc[0]
    print("\n================ Best Model by Mean F1 ================")
    print(best_model_row.to_string())

    print(f"\nDetailed results saved to: {os.path.abspath(detail_csv)}")
    print(f"Summary results saved to:   {os.path.abspath(summary_csv)}")

    # Visualization
    plot_comparison(summary_df, save_path=fig_bar)
    plot_repeat_f1(detail_df, save_path=fig_line)

    print(f"Comparison figure saved to: {os.path.abspath(fig_bar)}")
    print(f"Repeat F1 figure saved to:  {os.path.abspath(fig_line)}")


if __name__ == "__main__":
    main()