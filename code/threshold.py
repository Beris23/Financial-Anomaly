# Question 2: What threshold did you use to define an "anomaly," and how was it chosen?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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


print("="*60)
print("QUESTION 2: THRESHOLD SELECTION STRATEGIES")
print("="*60)

# Method 1: Percentile-based thresholds
print("\n1. PERCENTILE-BASED THRESHOLDS")
print("-" * 40)

percentiles = [75, 80, 85, 90, 95, 99]
thresholds = {}

for p in percentiles:
    threshold = np.percentile(data['Distance_to_Center'], p)
    num_anomalies = np.sum(data['Distance_to_Center'] > threshold)
    thresholds[p] = threshold
    print(f"{p} percentile: {threshold:.3f} -> {num_anomalies} anomalies ({num_anomalies/len(data)*100:.1f}%)")

# Method 2: Standard deviation-based thresholds
print("\n2. STANDARD DEVIATION-BASED THRESHOLDS")
print("-" * 40)

mean_dist = data['Distance_to_Center'].mean()
std_dist = data['Distance_to_Center'].std()

sd_multipliers = [1, 1.5, 2, 2.5, 3]
for multiplier in sd_multipliers:
    threshold = mean_dist + multiplier * std_dist
    num_anomalies = np.sum(data['Distance_to_Center'] > threshold)
    print(f"Mean + {multiplier}*SD: {threshold:.3f} -> {num_anomalies} anomalies ({num_anomalies/len(data)*100:.1f}%)")

# Method 3: Inter-Quartile Range (IQR) method
print("\n3. INTERQUARTILE RANGE (IQR) METHOD")
print("-" * 40)

Q1 = np.percentile(data['Distance_to_Center'], 25)
Q3 = np.percentile(data['Distance_to_Center'], 75)
IQR = Q3 - Q1

iqr_multipliers = [1.5, 2.0, 2.5, 3.0]
for multiplier in iqr_multipliers:
    threshold = Q3 + multiplier * IQR
    num_anomalies = np.sum(data['Distance_to_Center'] > threshold)
    print(f"Q3 + {multiplier}*IQR: {threshold:.3f} -> {num_anomalies} anomalies ({num_anomalies/len(data)*100:.1f}%)")

# Method 4: Fixed percentage approach
print("\n4. FIXED PERCENTAGE APPROACH")
print("-" * 40)
target_percentages = [5, 10, 15, 20]
for target_pct in target_percentages:
    threshold = np.percentile(data['Distance_to_Center'], 100 - target_pct)
    num_anomalies = np.sum(data['Distance_to_Center'] > threshold)
    print(f"Top {target_pct} percent: {threshold:.3f} -> {num_anomalies} anomalies")

# Let's choose the 90th percentile (top 10 percent) and analyze it
print("\n" + "="*60)
print("CHOSEN THRESHOLD: 90 PERCENTILE (TOP 10 PERCENT)")
print("="*60)

chosen_threshold = np.percentile(data['Distance_to_Center'], 90)
print(f"Threshold value: {chosen_threshold:.3f}")

# Identify anomalies
data['Is_Anomaly'] = data['Distance_to_Center'] > chosen_threshold
anomalies = data[data['Is_Anomaly']].copy()
normal_transactions = data[~data['Is_Anomaly']].copy()

print(f"\nNumber of anomalies detected: {len(anomalies)}")
print(f"Percentage of data flagged: {len(anomalies)/len(data)*100:.1f} percent")

print("\nDetected Anomalies:")
print(anomalies[['TransactionID', 'Name', 'Amount', 'Time', 'Cluster', 'Distance_to_Center']].to_string(index=False))

# WHY 90th percentile?
print("\n" + "="*60)
print("WHY CHOOSE 90TH PERCENTILE? (BUSINESS JUSTIFICATION)")
print("="*60)
print("""
REASONS FOR CHOOSING 90TH PERCENTILE (TOP 10%):

1. MANAGEABLE VOLUME: 10 anomalies out of 100 transactions (10%)
   - Not too many to overwhelm analysts
   - Not too few to miss important patterns
   
2. INDUSTRY STANDARD: 90TH percentile is commonly used in:
   - Credit card fraud detection
   - Network intrusion detection
   - Quality control systems
   
4. BALANCES PRECISION vs RECALL:
   - 95TH percentile (5%) might miss some real anomalies
   - 85TH percentile (15%) might flag too many false positives
   
3. STATISTICAL SIGNIFICANCE: 
   - Captures transactions genuinely far from normal behavior
   - Based on actual data distribution, not arbitrary cutoffs
   
5. OPERATIONAL FEASIBILITY:
   - 10% flagging rate is manageable for manual review
   - Allows for human verification before taking action
""")

# Visualize different thresholds
plt.figure(figsize=(15, 10))

# Plot 1: Distance distribution with different thresholds
plt.subplot(2, 3, 1)
plt.hist(data['Distance_to_Center'], bins=20, alpha=0.7, edgecolor='black')
plt.axvline(thresholds[85], color='blue', linestyle='--', label='85 percentile')
plt.axvline(thresholds[90], color='red', linestyle='--', label='90 percentile')
plt.axvline(thresholds[95], color='green', linestyle='--', label='95 percentile')
plt.xlabel('Distance to Cluster Center')
plt.ylabel('Frequency')
plt.title('Distance Distribution with Thresholds')
plt.legend()

# Plot 2: Number of anomalies vs threshold
plt.subplot(2, 3, 2)
threshold_values = [thresholds[p] for p in percentiles]
anomaly_counts = [np.sum(data['Distance_to_Center'] > t) for t in threshold_values]
plt.plot(percentiles, anomaly_counts, marker='o')
plt.axvline(90, color='red', linestyle='--', label='Chosen: 90 percentile')
plt.xlabel('Percentile Threshold')
plt.ylabel('Number of Anomalies')
plt.title('Anomalies vs Threshold')
plt.legend()

# Plot 3: Scatter plot with anomalies highlighted
plt.subplot(2, 3, 3)
plt.scatter(normal_transactions['Amount'], normal_transactions['Time'], 
           c='blue', alpha=0.6, label='Normal')
plt.scatter(anomalies['Amount'], anomalies['Time'], 
           c='red', alpha=0.8, s=100, label='Anomalies')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.title('Transactions with Anomalies Highlighted')
plt.legend()

# Plot 4: Distance vs Transaction ID
plt.subplot(2, 3, 4)
plt.plot(data['TransactionID'], data['Distance_to_Center'], 'o-', alpha=0.7)
plt.axhline(chosen_threshold, color='red', linestyle='--', label=f'Threshold: {chosen_threshold:.3f}')
plt.xlabel('Transaction ID')
plt.ylabel('Distance to Center')
plt.title('Distance Pattern Over Time')
plt.legend()

# Plot 5: Box plot by cluster
plt.subplot(2, 3, 5)
data.boxplot(column='Distance_to_Center', by='Cluster', ax=plt.gca())
plt.axhline(chosen_threshold, color='red', linestyle='--', label='Threshold')
plt.title('Distance Distribution by Cluster')
plt.suptitle('')  # Remove automatic title
plt.legend()

# Plot 6: Anomaly analysis by cluster
plt.subplot(2, 3, 6)
cluster_anomaly_counts = data.groupby('Cluster')['Is_Anomaly'].sum()
cluster_total_counts = data.groupby('Cluster').size()
anomaly_rates = cluster_anomaly_counts / cluster_total_counts * 100

plt.bar(range(len(cluster_anomaly_counts)), anomaly_rates)
plt.xlabel('Cluster')
plt.ylabel('Anomaly Rate (percent)')
plt.title('Anomaly Rate by Cluster')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("THRESHOLD SELECTION SUMMARY")
print("="*60)
print(f"""
CHOSEN METHOD: 90 Percentile
THRESHOLD VALUE: {chosen_threshold:.3f}
ANOMALIES DETECTED: {len(anomalies)}
FLAGGING RATE: {len(anomalies)/len(data)*100:.1f} percent

ALTERNATIVE APPROACHES:
- More Conservative (fewer false positives): 95 percentile -> {np.sum(data['Distance_to_Center'] > thresholds[95])} anomalies
- More Aggressive (catch more anomalies): 85 percentile -> {np.sum(data['Distance_to_Center'] > thresholds[85])} anomalies
- Statistical approach: Mean + 2*SD -> {np.sum(data['Distance_to_Center'] > mean_dist + 2*std_dist)} anomalies
""")

print("\n" + "="*60)
print("READY FOR QUESTION 3?")
print("="*60)
