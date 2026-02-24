import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture as GMM

df = pd.read_csv("pca_data.csv")

X = df[["PCA1", "PCA2"]]

gmm = GMM.GaussianMixture(n_components=2, covariance_type="full", random_state=42)
df["Cluster"] = gmm.fit_predict(X)

log_likelihood = gmm.score_samples(X)
threshold = np.percentile(log_likelihood, 5)
df["Anomaly"] = (log_likelihood < threshold).astype(int)

print(df["Cluster"].value_counts())
print(df["Anomaly"].value_counts())

plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis", label="Clusters")
plt.scatter(df[df["Anomaly"] == 1]["PCA1"], df[df["Anomaly"] == 1]["PCA2"], c="red", label="Anomalies", edgecolor="k")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("GMM Clustering and Anomaly Detection")
plt.legend()
plt.show()