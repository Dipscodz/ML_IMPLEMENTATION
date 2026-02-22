import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("preprocessed_data.csv")
X=df
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])

pca_df.to_csv("pca_data.csv", index=False)
