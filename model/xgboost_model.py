from xgboost import XGBClassifier

def get_model():
    return XGBClassifier(eval_metric="logloss")
