from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from data import load_data
import os
import pandas as pd
import joblib as jl
import json

def best_cv_split(X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_score = 0
    best_split = None
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        if score > best_score:
            best_score = score
            best_split = (train_index, test_index)

    if best_split is None:
        return None
    
    best_train_split = best_split[0]
    best_test_split = best_split[1]


    best_train_split_df = pd.DataFrame(X.iloc[best_train_split])
    best_test_split_df = pd.DataFrame(X.iloc[best_test_split])
    best_train_split_df['Churn'] = y.iloc[best_train_split].values

    best_train_split_df.to_csv(os.path.join('data',"processed", 'best_train_split.csv'), index=False)
    best_test_split_df.to_csv(os.path.join('data',"processed", 'best_test_split.csv'), index=False)

    return best_split

fstd = load_data(os.path.join('data', 'processed', 'feature_selected_train.csv'))
# best_cv_split(fstd.drop(columns=['Churn']), fstd['Churn'])

def hyperparameter_tuning(X, y):
    best_params = {}
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'ccp_alpha': [0.01, 0.05, 0.1]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    best_params["RandomForest"] = grid_search.best_params_

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'ccp_alpha': [0.01, 0.05, 0.1]
    }

    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    best_params["DecisionTree"] = grid_search.best_params_

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'min_impurity_decrease': [0.01, 0.05, 0.1],
        'max_leaf_nodes' : [None, 10, 20],
        'ccp_alpha': [0.01, 0.05, 0.1],
        'tol' : [0.01, 0.1]
    }
    model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    best_params["GradientBoosting"] = grid_search.best_params_

    return best_params

best_params = hyperparameter_tuning(fstd.drop(columns=['Churn']), fstd['Churn'])
json.dump(best_params, open(os.path.join('models', 'best_params.json'), 'w'))

def train_model(X, y):
    model = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(
            criterion='entropy',
            n_estimators=100, 
            max_depth=None, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            random_state=42)
        ),
        ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ], voting='hard')
    model.fit(X, y)
    return model