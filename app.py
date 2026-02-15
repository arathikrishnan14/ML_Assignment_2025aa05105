import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("ML Assignment 2: Classification Models")

# File uploader
data_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Handle missing values
    data = data.dropna()  # Drop rows with missing values
    # Alternatively, you can use imputation:
    # from sklearn.impute import SimpleImputer
    # imputer = SimpleImputer(strategy='mean')
    # X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Splitting dataset
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model_choice = st.selectbox("Choose a Model", [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbor",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ])

    # Initialize model
    model = None
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "K-Nearest Neighbor":
        model = KNeighborsClassifier()
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "XGBoost":
        model = XGBClassifier()

    # Train and evaluate
    if model is not None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        st.write("### Evaluation Metrics")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        if y_prob is not None:
            st.write(f"AUC: {roc_auc_score(y_test, y_prob):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"MCC: {matthews_corrcoef(y_test, y_pred):.2f}")

        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Classification Report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))
