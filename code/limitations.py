

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns


data = pd.read_csv("C:\\Users\\GIMBIYA BENJAMIN\\Desktop\\transactions_nigeria.csv")
X = data[['Amount', 'Time']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\nOVERVIEW: K-Means Assumptions vs Financial Data Reality")
print("-" * 50)
print("""
K-Means works well when data meets certain assumptions:
✓ Clusters are spherical (circular/round)
✓ Clusters are similar in size
✓ Clusters are well-separated
✓ No extreme outliers
✓ Features are on similar scales

But financial transaction data often violates these assumptions!
""")

print("\n" + "="*70)
print("LIMITATION 1: ASSUMES SPHERICAL CLUSTERS")
print("="*70)

print("""
PROBLEM:
- K-Means assumes clusters are circular/spherical
- Real financial patterns might be:
  * Linear (salary payments on specific dates)
  * Irregular shapes (seasonal spending patterns)
  * Elongated (transaction amounts following time trends)

IMPACT ON ANOMALY DETECTION:
- Normal transactions in non-spherical patterns get flagged as anomalies
- Algorithm forces round clusters even when data has other shapes
- Misclassifies edge cases in elongated or irregular patterns

REAL-WORLD EXAMPLE:
- Credit card payments: Often form linear patterns (same amount, different times)
- K-Means might split this into multiple round clusters
- Transactions at the "ends" get flagged as anomalous incorrectly
""")
np.random.seed(42)

elongated_x = np.random.normal(50, 5, 30)
elongated_y = elongated_x * 0.5 + np.random.normal(0, 2, 30) 
elongated_data = np.column_stack([elongated_x, elongated_y])

data = pd.read_csv("C:\\Users\\GIMBIYA BENJAMIN\\Desktop\\transactions_nigeria.csv")
real_data = data[['Amount', 'Time']].values[:20] 

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(elongated_data[:, 0], elongated_data[:, 1], alpha=0.6, label='Elongated Pattern')
kmeans_demo = KMeans(n_clusters=2, random_state=42)
labels = kmeans_demo.fit_predict(elongated_data)
centers = kmeans_demo.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='K-Means Centers')
plt.title('Problem: Forcing Round Clusters on Linear Data')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.6)
plt.title('Your Data: First 20 Transactions')
plt.xlabel('Amount')
plt.ylabel('Time')

plt.subplot(1, 3, 3)
outlier_demo = np.vstack([real_data[:15], [[1000, 500]]])  
kmeans_outlier = KMeans(n_clusters=3, random_state=42)
outlier_labels = kmeans_outlier.fit_predict(outlier_demo)
outlier_centers = kmeans_outlier.cluster_centers_
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_points = outlier_demo[outlier_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[i], alpha=0.6, label=f'Cluster {i}')
plt.scatter(outlier_centers[:, 0], outlier_centers[:, 1], 
           c='black', marker='x', s=200, label='Centers')
plt.title('Problem: Outliers Pull Centers Away')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("LIMITATION 2: SENSITIVE TO OUTLIERS")
print("="*70)

print("""
PROBLEM:
- Extreme outliers (like your transactions 96-100) pull cluster centers away
- This affects the classification of ALL other transactions
- Centers get "dragged" toward outliers, distorting normal patterns

IMPACT ON ANOMALY DETECTION:
- Normal transactions near outliers get misclassified
- Cluster centers don't represent "typical" behavior anymore
- Distance calculations become less meaningful

FROM YOUR DATA:
- Transactions 96-100 have amounts 300-1200 vs normal 30-70
- These extreme values pull cluster centers away from normal patterns
- Other normal transactions might get flagged incorrectly
""")

data_normal = data.iloc[:95]  
data_with_outliers = data.copy()  

scaler = StandardScaler()
X_normal = scaler.fit_transform(data_normal[['Amount', 'Time']])
X_with_outliers = scaler.fit_transform(data_with_outliers[['Amount', 'Time']])

kmeans_normal = KMeans(n_clusters=3, random_state=42)
kmeans_outliers = KMeans(n_clusters=3, random_state=42)

labels_normal = kmeans_normal.fit_predict(X_normal)
labels_outliers = kmeans_outliers.fit_predict(X_with_outliers)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data_normal['Amount'], data_normal['Time'], c=labels_normal, alpha=0.6, cmap='viridis')
centers_normal_orig = scaler.inverse_transform(kmeans_normal.cluster_centers_)
plt.scatter(centers_normal_orig[:, 0], centers_normal_orig[:, 1], 
           c='red', marker='x', s=200, label='Centers (No Outliers)')
plt.title('Clustering WITHOUT Extreme Outliers')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(data_with_outliers['Amount'], data_with_outliers['Time'], 
           c=labels_outliers, alpha=0.6, cmap='viridis')
centers_outliers_orig = scaler.inverse_transform(kmeans_outliers.cluster_centers_)
plt.scatter(centers_outliers_orig[:, 0], centers_outliers_orig[:, 1], 
           c='red', marker='x', s=200, label='Centers (With Outliers)')
plt.title('Clustering WITH Extreme Outliers')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.legend()

plt.tight_layout()
plt.show()

print(f"""
CENTER COMPARISON:
Without Outliers - Centers at: {centers_normal_orig.round(1)}
With Outliers - Centers at: {centers_outliers_orig.round(1)}

Notice how the outliers shifted the cluster centers!
""")

print("\n" + "="*70)
print("LIMITATION 3: REQUIRES PRE-SPECIFIED K (NUMBER OF CLUSTERS)")
print("="*70)

print("""
PROBLEM:
- You must decide how many clusters (k) to use BEFORE running the algorithm
- Wrong k leads to poor anomaly detection
- Financial patterns change over time - optimal k might change

IMPACT ON ANOMALY DETECTION:
- Too few clusters: Miss subtle anomaly patterns
- Too many clusters: Normal transactions get split, creating false anomalies
- No automatic way to find the "right" number of clusters

FINANCIAL CONTEXT:
- Transaction patterns vary by:
  * Time of day/week/month
  * User demographics  
  * Economic conditions
  * Seasonal factors
- Fixed k can't adapt to changing patterns
""")

from sklearn.metrics import silhouette_score

k_values = range(2, 8)
silhouette_scores = []
inertias = []

X = data[['Amount', 'Time']].values
scaler_demo = StandardScaler()
X_scaled_demo = scaler_demo.fit_transform(X)

for k in k_values:
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans_k.fit_predict(X_scaled_demo)
    silhouette_avg = silhouette_score(X_scaled_demo, labels_k)
    silhouette_scores.append(silhouette_avg)
    inertias.append(kmeans_k.inertia_)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_values, inertias, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Silhouette scores for different k values: {dict(zip(k_values, [round(s, 3) for s in silhouette_scores]))}")

print("\n" + "="*70)
print("LIMITATION 4: IGNORES CONTEXT AND TEMPORAL PATTERNS")
print("="*70)

print("""
PROBLEM:
- K-Means only looks at Amount and Time as isolated values
- Ignores important context like:
  * User spending history
  * Time-of-day patterns (rush hour transactions)
  * Day-of-week patterns (weekend vs weekday)
  * Sequential patterns (repeated transactions)
  * Seasonal trends

IMPACT ON ANOMALY DETECTION:
- A ₦100,000 transaction might be:
  * Normal for a business account
  * Highly anomalous for a student account
- Same transaction amount at different times might have different risk levels
- Algorithm can't learn from user behavior history

MISSING CONTEXT EXAMPLES:
- User profile: Age, income, location, account type
- Transaction sequence: Is this part of a pattern?
- Temporal context: Holiday season, salary day, etc.
- Merchant context: ATM vs online vs POS
""")

print("\n" + "="*70)
print("LIMITATION 5: SCALE SENSITIVITY")
print("="*70)

print("""
PROBLEM:
- Features with larger scales dominate the distance calculation
- In your data: Amount (30-1200) vs Time (20-500)
- Without proper scaling, Amount differences overwhelm Time differences

IMPACT ON ANOMALY DETECTION:
- Algorithm might only focus on amount anomalies, missing time anomalies
- Relative importance of features gets distorted
- Need careful preprocessing and domain knowledge

FROM YOUR DATA:
""")

X_unscaled = data[['Amount', 'Time']].values
X_scaled = StandardScaler().fit_transform(X_unscaled)

print(f"Original data ranges:")
print(f"Amount: {X_unscaled[:, 0].min():.1f} to {X_unscaled[:, 0].max():.1f} (range: {X_unscaled[:, 0].max() - X_unscaled[:, 0].min():.1f})")
print(f"Time: {X_unscaled[:, 1].min():.1f} to {X_unscaled[:, 1].max():.1f} (range: {X_unscaled[:, 1].max() - X_unscaled[:, 1].min():.1f})")

print(f"\nScaled data ranges:")
print(f"Amount: {X_scaled[:, 0].min():.1f} to {X_scaled[:, 0].max():.1f}")
print(f"Time: {X_scaled[:, 1].min():.1f} to {X_scaled[:, 1].max():.1f}")

print("\n" + "="*70)
print("LIMITATION 6: NO DENSITY-BASED ANOMALY DETECTION")
print("="*70)

print("""
PROBLEM:
- K-Means assigns EVERY point to a cluster, even clear outliers
- Can't identify "noise" points that don't belong to any cluster
- Distance-based approach misses density-based anomalies

IMPACT ON ANOMALY DETECTION:
- Isolated points get forced into nearest cluster
- Sparse regions (low-density areas) not automatically flagged
- Can't distinguish between "edge of cluster" vs "completely isolated"

BETTER ALTERNATIVES:
- DBSCAN: Identifies noise points automatically
- Isolation Forest: Designed specifically for anomaly detection
- Local Outlier Factor: Considers local density
""")

print("\n" + "="*70)
print("COMPREHENSIVE SUMMARY OF LIMITATIONS")
print("="*70)

limitations_summary = {
    "Spherical Clusters": "Forces round clusters on non-round data patterns",
    "Outlier Sensitivity": "Extreme values distort all cluster centers",
    "Fixed K Requirement": "Must pre-specify number of clusters",
    "Context Ignorance": "Ignores user history, temporal patterns, domain knowledge",
    "Scale Sensitivity": "Features with large ranges dominate distance calculations",
    "No Density Detection": "Forces all points into clusters, even clear outliers",
    "Static Approach": "Can't adapt to evolving transaction patterns",
    "Binary Classification": "Only 'normal' or 'anomalous', no confidence scores"
}

print("\nLIMITATION DETAILS:")
for i, (limitation, description) in enumerate(limitations_summary.items(), 1):
    print(f"{i}. {limitation.upper()}")
    print(f"   └─ {description}")

print("\n" + "="*70)
print("BETTER ALTERNATIVES FOR FINANCIAL ANOMALY DETECTION")
print("="*70)

alternatives = {
    "Isolation Forest": "Designed for anomaly detection, handles mixed data types",
    "DBSCAN": "Density-based clustering, automatically identifies noise",
    "Local Outlier Factor": "Considers local neighborhood density",
    "One-Class SVM": "Learns normal behavior boundary",
    "Autoencoders": "Neural networks that learn normal patterns",
    "Ensemble Methods": "Combine multiple algorithms for robustness",
    "Rule-Based Systems": "Incorporate domain expertise and business rules",
    "Time Series Analysis": "Handles temporal dependencies and seasonality"
}

print("RECOMMENDED ALTERNATIVES:")
for i, (method, description) in enumerate(alternatives.items(), 1):
    print(f"{i}. {method.upper()}")
    print(f"   └─ {description}")

print("\n" + "="*70)
print("PRACTICAL RECOMMENDATIONS")
print("="*70)

print("""
FOR PRODUCTION FINANCIAL ANOMALY DETECTION:

1. HYBRID APPROACH
   └─ Use K-Means as ONE component in a larger system
   └─ Combine with rule-based checks and domain expertise
   └─ Add user context and historical patterns

2. PREPROCESSING IMPORTANCE  
   └─ Careful feature scaling and normalization
   └─ Handle extreme outliers before clustering
   └─ Feature engineering (ratios, moving averages, etc.)

3. ADAPTIVE SYSTEMS
   └─ Regularly retrain models as patterns evolve
   └─ Use different models for different user segments
   └─ Implement feedback loops from fraud analysts

4. MULTIPLE VALIDATION LAYERS
   └─ Statistical tests (like we did with percentiles)
   └─ Business rule validation
   └─ Human expert review for high-risk cases
   └─ Customer verification for suspicious transactions

5. PERFORMANCE MONITORING
   └─ Track false positive rates
   └─ Monitor detection accuracy over time  
   └─ A/B test different threshold strategies
   └─ Measure business impact (fraud prevented vs customer friction)
""")

print("\n" + "="*70)
print("FINAL ANSWER SUMMARY")
print("="*70)

print("""
K-MEANS LIMITATIONS FOR FINANCIAL ANOMALY DETECTION:

✗ ASSUMES SPHERICAL CLUSTERS - Financial patterns aren't always circular
✗ SENSITIVE TO OUTLIERS - Extreme transactions distort normal patterns  
✗ REQUIRES FIXED K - Can't adapt to changing transaction patterns
✗ IGNORES CONTEXT - Misses user history, time patterns, domain knowledge
✗ SCALE DEPENDENT - Large amount ranges can overwhelm time patterns
✗ NO DENSITY AWARENESS - Forces outliers into clusters instead of flagging them
✗ STATIC APPROACH - Can't evolve with changing fraud patterns
✗ LIMITED INTERPRETABILITY - Hard to explain "why" to business users

DESPITE LIMITATIONS:
✓ Good starting point for understanding data patterns
✓ Computationally efficient for large datasets
✓ Provides interpretable cluster centers
✓ Works well when combined with other methods
✓ Useful for initial data exploration and feature engineering

RECOMMENDATION: Use K-Means as part of a comprehensive anomaly detection
system, not as the sole method for financial fraud detection.
""")

with open("output.txt", "w", encoding="utf-8") as f:
    import sys
    sys.stdout = f
