
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns

data = pd.read_csv("C:\\Users\\GIMBIYA BENJAMIN\\Desktop\\transactions_nigeria.csv")
X = data[['Amount', 'Time']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
centers_scaled = kmeans.cluster_centers_

def calculate_distances_to_centers(X_scaled, centers_scaled, labels):
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

print("WHAT ARE BOUNDARY TRANSACTIONS?")
print("-" * 40)
print("""
Boundary transactions are those that:
1. Are close to the edge between two or more clusters
2. Could reasonably belong to multiple clusters
3. Have ambiguous cluster assignments
4. May be misclassified due to their position

PROBLEM: These transactions might be:
- Falsely flagged as anomalies when they're just on cluster edges
- Actually normal but appear anomalous due to poor cluster fit
""")
print("\n1. DISTANCE RATIO ANALYSIS")
print("-" * 40)

def calculate_distance_ratios(X_scaled, centers_scaled, labels):
    """
    Calculate ratio of distance to second-closest center vs closest center
    Low ratio = close to boundary (ambiguous assignment)
    High ratio = clearly belongs to assigned cluster
    """
    ratios = []
    second_distances = []
    closest_distances = []
    
    for i in range(len(X_scaled)):
        distances_to_all = []
        for center in centers_scaled:
            dist = np.sqrt(np.sum((X_scaled[i] - center) ** 2))
            distances_to_all.append(dist)
        
        sorted_distances = sorted(distances_to_all)
        closest_dist = sorted_distances[0]
        second_closest_dist = sorted_distances[1]
        
        ratio = second_closest_dist / closest_dist
        
        ratios.append(ratio)
        closest_distances.append(closest_dist)
        second_distances.append(second_closest_dist)
    
    return np.array(ratios), np.array(closest_distances), np.array(second_distances)

ratios, closest_dists, second_dists = calculate_distance_ratios(X_scaled, centers_scaled, cluster_labels)

data['Distance_Ratio'] = ratios
data['Closest_Distance'] = closest_dists
data['Second_Closest_Distance'] = second_dists

print("Distance Ratio Statistics:")
print(f"Mean ratio: {ratios.mean():.3f}")
print(f"Std ratio: {ratios.std():.3f}")
print(f"Min ratio: {ratios.min():.3f}")
print(f"Max ratio: {ratios.max():.3f}")

BOUNDARY_THRESHOLD = 1.3  # If ratio < 1.3, consider it a boundary case
boundary_transactions = data[data['Distance_Ratio'] < BOUNDARY_THRESHOLD].copy()

print(f"\nBoundary transactions (ratio < {BOUNDARY_THRESHOLD}): {len(boundary_transactions)}")
print(f"Percentage of transactions on boundaries: {len(boundary_transactions)/len(data)*100:.1f} percent")

if len(boundary_transactions) > 0:
    print("\nBoundary Transactions:")
    print(boundary_transactions[['TransactionID', 'Name', 'Amount', 'Time', 'Cluster', 'Distance_Ratio']].head(10))

print("\n2. SILHOUETTE ANALYSIS")
print("-" * 40)

silhouette_avg = silhouette_score(X_scaled, cluster_labels)
silhouette_values = silhouette_samples(X_scaled, cluster_labels)

data['Silhouette_Score'] = silhouette_values

print(f"Average Silhouette Score: {silhouette_avg:.3f}")
print("Silhouette Score Interpretation:")
print("- Close to +1: Well separated from neighboring clusters")
print("- Close to 0: On or very close to decision boundary")
print("- Negative: May have been assigned to wrong cluster")

SILHOUETTE_THRESHOLD = 0.2
poor_clustering = data[data['Silhouette_Score'] < SILHOUETTE_THRESHOLD].copy()

print(f"\nPoorly clustered transactions (silhouette < {SILHOUETTE_THRESHOLD}): {len(poor_clustering)}")

if len(poor_clustering) > 0:
    print("\nPoorly Clustered Transactions:")
    print(poor_clustering[['TransactionID', 'Name', 'Amount', 'Time', 'Cluster', 'Silhouette_Score']].head(10))

print("\n3. COMBINED BOUNDARY DETECTION")
print("-" * 40)

def classify_transaction_confidence(row):
    """
    Classify transactions based on multiple criteria
    """
    ratio = row['Distance_Ratio']
    silhouette = row['Silhouette_Score']
    distance = row['Distance_to_Center']
    
    if ratio < 1.2 and silhouette < 0.1:
        return 'High_Boundary_Risk'
    elif ratio < 1.3 or silhouette < 0.2:
        return 'Moderate_Boundary_Risk'
    elif distance > np.percentile(data['Distance_to_Center'], 90):
        return 'Potential_Anomaly'
    else:
        return 'Confident_Assignment'

data['Confidence_Level'] = data.apply(classify_transaction_confidence, axis=1)

confidence_counts = data['Confidence_Level'].value_counts()
print("Transaction Confidence Classification:")
for conf_level, count in confidence_counts.items():
    print(f"- {conf_level}: {count} transactions ({count/len(data)*100:.1f} percent)")

print("\n4. BOUNDARY HANDLING STRATEGIES")
print("-" * 40)

print("""
STRATEGY 1: CONSERVATIVE FLAGGING
- Only flag transactions as anomalies if they're clearly far from ALL clusters
- Require higher threshold for boundary transactions
- Reduces false positives but might miss some real anomalies

STRATEGY 2: CONTEXT-AWARE ANALYSIS  
- Consider boundary transactions separately
- Use additional features (time patterns, user history, etc.)
- Manual review for ambiguous cases

STRATEGY 3: ENSEMBLE APPROACH
- Use multiple clustering algorithms (K-means, DBSCAN, etc.)
- Flag only if multiple methods agree
- More robust but computationally expensive

STRATEGY 4: ADAPTIVE THRESHOLDS
- Lower threshold for confident assignments
- Higher threshold for boundary cases
- Balances sensitivity with specificity
""")

print("\n5. IMPLEMENTING ADAPTIVE THRESHOLDS")
print("-" * 40)

base_threshold = np.percentile(data['Distance_to_Center'], 90)

def adaptive_anomaly_detection(row):
    """
    Apply different thresholds based on confidence level
    """
    distance = row['Distance_to_Center']
    confidence = row['Confidence_Level']
    
    if confidence == 'Confident_Assignment':
        threshold = base_threshold
    elif confidence == 'Moderate_Boundary_Risk':
        threshold = base_threshold * 1.2  # 20 percent higher threshold
    elif confidence == 'High_Boundary_Risk':
        threshold = base_threshold * 1.5  # 50 percent higher threshold
    else:  # Potential_Anomaly
        threshold = base_threshold
    
    return distance > threshold

data['Adaptive_Anomaly'] = data.apply(adaptive_anomaly_detection, axis=1)

#we compare the result here btw simple and adaptive
simple_anomalies = data[data['Distance_to_Center'] > base_threshold]
adaptive_anomalies = data[data['Adaptive_Anomaly']]

print(f"Simple threshold anomalies: {len(simple_anomalies)}")
print(f"Adaptive threshold anomalies: {len(adaptive_anomalies)}")
print(f"Difference: {len(simple_anomalies) - len(adaptive_anomalies)} transactions")

print("\nAdaptive Anomaly Results:")
if len(adaptive_anomalies) > 0:
    print(adaptive_anomalies[['TransactionID', 'Name', 'Amount', 'Time', 'Confidence_Level']].to_string(index=False))

plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.hist(data['Distance_Ratio'], bins=20, alpha=0.7, edgecolor='black')
plt.axvline(BOUNDARY_THRESHOLD, color='red', linestyle='--', label=f'Boundary threshold: {BOUNDARY_THRESHOLD}')
plt.xlabel('Distance Ratio (2nd closest / closest)')
plt.ylabel('Frequency')
plt.title('Distance Ratio Distribution')
plt.legend()

plt.subplot(2, 3, 2)
plt.hist(data['Silhouette_Score'], bins=20, alpha=0.7, edgecolor='black')
plt.axvline(SILHOUETTE_THRESHOLD, color='red', linestyle='--', label=f'Poor clustering threshold: {SILHOUETTE_THRESHOLD}')
plt.xlabel('Silhouette Score')
plt.ylabel('Frequency')
plt.title('Silhouette Score Distribution')
plt.legend()

plt.subplot(2, 3, 3)
colors = {'Confident_Assignment': 'green', 'Moderate_Boundary_Risk': 'yellow', 
          'High_Boundary_Risk': 'red', 'Potential_Anomaly': 'purple'}
for conf_level in data['Confidence_Level'].unique():
    subset = data[data['Confidence_Level'] == conf_level]
    plt.scatter(subset['Amount'], subset['Time'], 
               c=colors.get(conf_level, 'gray'), 
               label=conf_level, alpha=0.7)
plt.xlabel('Amount')
plt.ylabel('Time')
plt.title('Transactions by Confidence Level')
plt.legend()

#  Distance vs Ratio
plt.subplot(2, 3, 4)
scatter = plt.scatter(data['Distance_to_Center'], data['Distance_Ratio'], 
                     c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Distance to Center')
plt.ylabel('Distance Ratio')
plt.title('Distance vs Ratio (colored by cluster)')
plt.colorbar(scatter)

# Silhouette vs Distance Ratio
plt.subplot(2, 3, 5)
plt.scatter(data['Distance_Ratio'], data['Silhouette_Score'], alpha=0.6)
plt.axhline(SILHOUETTE_THRESHOLD, color='red', linestyle='--', alpha=0.5)
plt.axvline(BOUNDARY_THRESHOLD, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Distance Ratio')
plt.ylabel('Silhouette Score')
plt.title('Boundary Risk Assessment')

# Comparison of Anomaly Detection Methods
plt.subplot(2, 3, 6)
comparison_data = {
    'Simple Threshold': len(simple_anomalies),
    'Adaptive Threshold': len(adaptive_anomalies),
    'Boundary Cases': len(boundary_transactions),
    'Poor Clustering': len(poor_clustering)
}
plt.bar(comparison_data.keys(), comparison_data.values())
plt.ylabel('Number of Transactions')
plt.title('Anomaly Detection Method Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("BOUNDARY HANDLING SUMMARY")
print("="*60)
print(f"""
KEY FINDINGS:

1. BOUNDARY TRANSACTIONS: {len(boundary_transactions)} transactions ({len(boundary_transactions)/len(data)*100:.1f} percent) 
   are close to cluster boundaries (distance ratio < {BOUNDARY_THRESHOLD})

2. POOR CLUSTERING: {len(poor_clustering)} transactions ({len(poor_clustering)/len(data)*100:.1f} percent)
   have low silhouette scores (< {SILHOUETTE_THRESHOLD})

3. ADAPTIVE APPROACH: Reduced false positives by {len(simple_anomalies) - len(adaptive_anomalies)} transactions
   while maintaining detection of clear anomalies

RECOMMENDATIONS:
- Use adaptive thresholds for boundary cases
- Manual review for high boundary risk transactions  
- Consider additional features for ambiguous cases
- Regular model retraining as data patterns evolve
""")

print("\n" + "="*60)
print("READY FOR QUESTION 4?")
print("="*60)
print("Next, we'll explore the limitations of K-Means for financial anomaly detection!")