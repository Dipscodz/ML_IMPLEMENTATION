import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('classifier', RandomForestClassifier())
])


df = pd.read_csv("spaceship-titanic/train.csv")
df = df.drop(columns=["Transported"])
df["Destination"] = df["Destination"].fillna("1,9000")
df["HomePlanet"] = df["HomePlanet"].fillna("1,9000")

scaler = StandardScaler()
destination_scaler = scaler.fit_transform(df[["Destination"]])
home_planet_scaler = scaler.fit_transform(df[["HomePlanet"]])
df["Destination"] = destination_scaler
df["HomePlanet"] = home_planet_scaler





print(df.head())




