

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data = pd.read_csv("C:\\Users\\GIMBIYA BENJAMIN\\Desktop\\transactions_nigeria.csv")
print("Data loaded successfully!")
print(f"Shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

print("\nBasic statistics:")
print(data.describe())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data['Amount'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(data['Time'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Transaction Times')
plt.xlabel('Time')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

X = data[['Amount', 'Time']].values
print(f"\nFeature matrix shape: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nOriginal data range:")
print(f"Amount: {X[:, 0].min():.2f} to {X[:, 0].max():.2f}")
print(f"Time: {X[:, 1].min():.2f} to {X[:, 1].max():.2f}")

print("\nScaled data range:")
print(f"Amount: {X_scaled[:, 0].min():.2f} to {X_scaled[:, 0].max():.2f}")
print(f"Time: {X_scaled[:, 1].min():.2f} to {X_scaled[:, 1].max():.2f}")

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)


centers_scaled = kmeans.cluster_centers_

centers_original = scaler.inverse_transform(centers_scaled)

print(f"\nK-Means completed with {k} clusters")
print("\nCluster centers (original scale):")
for i, center in enumerate(centers_original):
    print(f"Cluster {i}: Amount={center[0]:.2f}, Time={center[1]:.2f}")

def calculate_distances_to_centers(X_scaled, centers_scaled, labels):
    """
    Calculate the distance from each point to its assigned cluster center
    """
    distances = []
    for i in range(len(X_scaled)):
        cluster_id = labels[i]
        center = centers_scaled[cluster_id]
        distance = np.sqrt(np.sum((X_scaled[i] - center) ** 2))
        distances.append(distance)
    
    return np.array(distances)


distances = calculate_distances_to_centers(X_scaled, centers_scaled, cluster_labels)

data['Cluster'] = cluster_labels
data['Distance_to_Center'] = distances

print("\nDistance statistics:")
print(f"Mean distance: {distances.mean():.3f}")
print(f"Std distance: {distances.std():.3f}")
print(f"Min distance: {distances.min():.3f}")
print(f"Max distance: {distances.max():.3f}")
print("\nTop 10 transactions with highest distances to their cluster centers:")
top_distances = data.nlargest(10, 'Distance_to_Center')[['TransactionID', 'Name', 'Amount', 'Time', 'Cluster', 'Distance_to_Center']]
print(top_distances)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(data['Amount'], data['Time'], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(centers_original[:, 0], centers_original[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.xlabel('Amount')
plt.ylabel('Time')
plt.title('Clusters (Original Space)')
plt.colorbar(scatter)

plt.subplot(1, 3, 2)
plt.hist(distances, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Distance to Cluster Center')
plt.ylabel('Frequency')
plt.title('Distribution of Distances')
plt.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.3f}')
plt.legend()

plt.subplot(1, 3, 3)
for cluster_id in range(k):
    cluster_mask = cluster_labels == cluster_id
    plt.scatter(data.loc[cluster_mask, 'Amount'], 
               data.loc[cluster_mask, 'Time'], 
               c=distances[cluster_mask], 
               cmap='Reds', 
               alpha=0.7,
               label=f'Cluster {cluster_id}')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.title('Transactions colored by Distance to Center')
plt.colorbar(label='Distance')

plt.tight_layout()
plt.show()

