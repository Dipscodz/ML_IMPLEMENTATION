import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv("pca_data.csv")
X = df[["PCA1", "PCA2"]]


kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)




distances = kmeans.transform(X)


min_distance = np.min(distances, axis=1)

threshold = np.percentile(min_distance, 95)

df["Anomaly"] = (min_distance > threshold).astype(int)


print(df["Cluster"].value_counts())
print(df["Anomaly"].value_counts())


plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis", label="Clusters")
plt.scatter(df[df["Anomaly"] == 1]["PCA1"],
            df[df["Anomaly"] == 1]["PCA2"],
            c="red", label="Anomalies", edgecolor="k")

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("KMeans Clustering and Anomaly Detection")
plt.legend()
plt.show()