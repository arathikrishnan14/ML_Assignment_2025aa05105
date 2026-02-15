import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

from model.logistic_model import get_model as logistic_model
from model.decision_tree_model import get_model as dt_model
from model.knn_model import get_model as knn_model
from model.naive_bayes_model import get_model as nb_model
from model.random_forest_model import get_model as rf_model
from model.xgboost_model import get_model as xgb_model

import matplotlib.pyplot as plt
import seaborn as sns

st.title("ML Assignment 2 — Classification Models Demo")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # assume last column is target
    df = df.dropna(subset=[df.columns[-1]])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # encode categorical features
    if X.select_dtypes(include="object").shape[1] > 0:
        X = pd.get_dummies(X)

    # impute missing
    imp = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": logistic_model(),
        "Decision Tree": dt_model(),
        "KNN": knn_model(),
        "Naive Bayes": nb_model(),
        "Random Forest": rf_model(),
        "XGBoost": xgb_model()
    }

    choice = st.selectbox("Choose Model", list(models.keys()))
    model = models[choice]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    st.subheader("Evaluation Metrics")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)
    st.write("MCC:", mcc)

    if y_prob is not None:
        if len(np.unique(y)) > 2:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        else:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        st.write("AUC:", auc)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # -------- comparison table --------

    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)

        if hasattr(m, "predict_proba"):
            prob = m.predict_proba(X_test)
            if len(np.unique(y)) > 2:
                auc = roc_auc_score(y_test, prob, multi_class="ovr")
            else:
                auc = roc_auc_score(y_test, prob[:, 1])
        else:
            auc = np.nan

        rows.append([
            name,
            accuracy_score(y_test, p),
            auc,
            precision_score(y_test, p, average="weighted"),
            recall_score(y_test, p, average="weighted"),
            f1_score(y_test, p, average="weighted"),
            matthews_corrcoef(y_test, p)
        ])

    comp = pd.DataFrame(rows, columns=[
        "Model","Accuracy","AUC","Precision","Recall","F1","MCC"
    ])

    st.subheader("All Model Comparison")
    st.dataframe(comp)
