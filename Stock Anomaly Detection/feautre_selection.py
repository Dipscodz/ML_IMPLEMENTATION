import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

df = pd.read_csv("preprocessed_data.csv")
X=df

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])

pca_df.to_csv("pca_data.csv", index=False)

plt.scatter(pca_df["PCA1"], pca_df["PCA2"])
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA of Stock Data")
plt.show()
