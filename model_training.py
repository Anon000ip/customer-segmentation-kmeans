import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Model training script for Customer Segmentation
def train_kmeans():
    X = np.random.rand(100, 2)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    print("Model trained successfully.")

if __name__ == "__main__":
    train_kmeans()
