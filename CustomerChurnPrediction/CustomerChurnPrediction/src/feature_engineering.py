from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

def feature_engineering_pipeline():
    model = LogisticRegression()
    pipeline = Pipeline([
        ('feature_selector', SFS(estimator=model, n_features_to_select='auto', direction='forward', scoring="accuracy", cv=5)),
    ])
    return pipeline