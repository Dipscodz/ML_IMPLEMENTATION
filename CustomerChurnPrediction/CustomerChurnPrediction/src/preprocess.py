from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

def categorical_pipeline():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown='ignore')),
    ])
    
    return pipeline

def numerical_pipeline():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler())
    ])
    
    return pipeline

def main_pipeline():
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', numerical_pipeline(), ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']),
            ('cat', ColumnTransformer(transformers=[
                ("onehot", OneHotEncoder(handle_unknown='ignore'), ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']),
                ("ordinal", OrdinalEncoder(), ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'Churn'])
            ]), ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'])
        ]))
    ])

    return pipeline

def test_pipeline():
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', numerical_pipeline(), ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']),
            ('cat', ColumnTransformer(transformers=[
                ("onehot", OneHotEncoder(handle_unknown='ignore'), ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']),
                ("ordinal", OrdinalEncoder(), ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'])
            ]), ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'])
        ]))
    ])

    return pipeline