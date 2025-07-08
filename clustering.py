# clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(df_cleaned, scaled_data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df_cleaned["Cluster"] = clusters
    profile = df_cleaned.groupby("Cluster").mean(numeric_only=True).round(2)
    return df_cleaned, profile

