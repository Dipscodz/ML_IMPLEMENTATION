import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("pca_data.csv")

X = df[["PCA1", "PCA2"]]

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

df["Cluster"] = kmeans.predict(X)

log_likelihood = kmeans.score_samples(X)
threshold = np.percentile(log_likelihood, 5)
df["Anomaly"] = (log_likelihood < threshold).astype(int)

print(df["Cluster"].value_counts())
print(df["Anomaly"].value_counts())

plt.figure(figsize=(8,6))
plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis", alpha=0.6)

plt.scatter(df[df["Anomaly"] == 1]["PCA1"],
df[df["Anomaly"] == 1]["PCA2"],
color="red", edgecolors="black")

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("GMM Clustering + Anomaly Detection")
plt.show()