# -*- coding: utf-8 -*-
"""
Improved bug report classification experiments.

This script compares three stronger models against the baseline idea:
1) Logistic Regression + TF-IDF
2) Linear SVM + Word/Char TF-IDF
3) Complement Naive Bayes + TF-IDF

It performs repeated train/test splits, grid search tuning, and saves both
detailed per-run results and summary statistics.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd

# Suppress all warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

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
from sklearn.naive_bayes import ComplementNB


# =========================================================
# 1. Global configuration
# =========================================================

RANDOM_STATE = 42
REPEAT = 5

project = "pytorch"
datafile = "Title+Body.csv"

detail_csv = f"./{project}_improved_detailed_results.csv"
summary_csv = f"./{project}_improved_summary_results.csv"


# =========================================================
# 2. Text preprocessing
# =========================================================

def remove_html(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r"<.*?>", " ", str(text))


def remove_emoji(text: str) -> str:
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", str(text))


def normalize_text(text: str) -> str:
    """
    Normalize text for software engineering bug reports.
    The cleaning keeps useful programming-related symbols and tokens.
    """
    text = str(text)
    text = remove_html(text)
    text = remove_emoji(text)

    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"`[^`]*`", " CODE ", text)
    text = re.sub(r"```.*?```", " CODEBLOCK ", text, flags=re.S)

    text = re.sub(r"[^A-Za-z0-9_+\-#/.:!? ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset and preprocess the text column.
    Required columns:
    - text
    - sentiment
    """
    df = pd.read_csv(csv_path).fillna("")

    required_cols = {"text", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["text"] = df["text"].apply(normalize_text)
    return df


# =========================================================
# 3. Metric computation
# =========================================================

def get_average_type(y_true: np.ndarray) -> str:
    """Use binary averaging for binary labels, macro otherwise."""
    n_classes = len(np.unique(y_true))
    return "binary" if n_classes == 2 else "macro"


def get_auc_score(estimator, X_test, y_test, n_classes: int):
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


def evaluate_model(estimator, X_test, y_test, n_classes: int):
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


# =========================================================
# 4. Model builders
# =========================================================

def build_logreg_pipeline():
    """
    Logistic Regression with TF-IDF.
    Improvements:
    - strong linear classifier for sparse text data
    - class balancing option
    - n-gram tuning
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
    Improvements:
    - word-level features capture semantics
    - char-level features capture spelling variants, code tokens, and noise robustness
    - dual is set explicitly to suppress the FutureWarning
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
    Improvements:
    - more suitable than GaussianNB for sparse text features
    - fast and effective on imbalanced text classification
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
    "LogReg_TFIDF": build_logreg_pipeline,
    "LinearSVM_WordCharTFIDF": build_linearsvm_pipeline,
    "ComplementNB_TFIDF": build_complementnb_pipeline,
}


# =========================================================
# 5. Training and tuning
# =========================================================

def run_one_model(
    model_name: str,
    pipeline,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    n_classes: int,
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


def main():
    # Load and preprocess data
    data = load_data(datafile)

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
                f"  CV_F1={result['Best_CV_F1']:.4f} | "
                f"Acc={result['Accuracy']:.4f} | "
                f"F1={result['F1']:.4f} | "
                f"AUC={result['AUC']:.4f}"
            )

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

    print("\n================ Overall Summary ================")
    print(summary_df.to_string(index=False))

    print(f"\nDetailed results saved to: {os.path.abspath(detail_csv)}")
    print(f"Summary results saved to:   {os.path.abspath(summary_csv)}")


if __name__ == "__main__":
    main()