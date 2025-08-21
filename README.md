 Project Overview


This project implements and evaluates K-Means clustering for detecting anomalous transactions in a dataset of 100 Nigerian financial transactions. The analysis covers threshold selection, boundary transaction handling, and discusses the limitations of K-Means for financial anomaly detection.
 Dataset

Source: Nigerian transaction data (transactions_nigeria.csv)
Size: 100 transactions
Features:

TransactionID: Unique identifier
Name: Customer name
Amount: Transaction amount (₦)
Time: Transaction time


Characteristics: Mix of normal transactions (₦30-70) and extreme outliers (₦300-1200)

 Key Findings
Anomaly Detection Results

Method: 90th percentile distance threshold
Anomalies Detected: 10 transactions (10% of dataset)
Primary Outliers: Transactions 96-100 with amounts 5-20x normal values
Clustering Quality: Silhouette score of 0.945 (excellent separation)

Critical Anomalies Identified

Transaction 100 (Mary): ₦1,200 - 20x median amount
Transaction 99 (Yemi): ₦1,000 - 17x median amount
Transaction 98 (Ebuka): ₦700 - 12x median amount
Transaction 97 (Onyeka): ₦500 - 8x median amount
Transaction 96 (Chiamaka): ₦300 - 5x median amount

 Technical Implementation
Methodology

Data Preprocessing: StandardScaler normalization
Clustering: K-Means with k=3 clusters
Distance Calculation: Euclidean distance to cluster centers
Threshold Selection: 90th percentile of distance distribution
Boundary Analysis: Distance ratio and silhouette score evaluation

Why 90th Percentile Threshold?

Manageable volume: 10% flagging rate for manual review
Industry standard: Common in fraud detection systems
Balance: Minimizes false positives while catching genuine anomalies
Statistical significance: Data-driven rather than arbitrary cutoff


 Getting Started
Prerequisites
bashpip install pandas numpy matplotlib scikit-learn seaborn
Running the Analysis
python# Run complete analysis
python src/final_anomaly_detection.py

# Individual components
python src/question1_distance_analysis.py
python src/question2_threshold_selection.py
python src/question3_boundary_handling.py
python src/question4_limitations.py
 Methodology Deep Dive
How distance to cluster centers flags anomalies:

Calculate Euclidean distance from each transaction to its assigned cluster center
Transactions with high distances are considered anomalous
Formula: distance = √[(amount₁ - center_amount)² + (time₁ - center_time)

-Multiple threshold methods evaluated:
90th Percentile (chosen): Balanced approach, 10% flagging rate
Statistical (Mean + 2σ): More conservative, fewer false positives
IQR Method (Q3 + 1.5×IQR): Robust to extreme outliers
Business Rules: Domain-specific thresholds (e.g., 3× median amount)

-Addressing ambiguous cluster assignments:

Distance Ratio Analysis: Compare distances to multiple cluster centers
Silhouette Scores: Measure cluster assignment confidence
Adaptive Thresholds: Higher thresholds for boundary cases
Result: Reduced false positives while maintaining detection accuracy

-Critical limitations for financial data:

Assumes spherical clusters (financial patterns may be non-circular)
Sensitive to extreme outliers (affects cluster centers)
Requires predetermined k (can't adapt to changing patterns)
Ignores temporal context and user behavior history
Scale sensitivity between different feature ranges

 Results Summary
Clustering Performance
Cluster 0: 34 transactions (Amount: ₦47.2±8.1, Time: 29.4±4.8)
Cluster 1: 31 transactions (Amount: ₦51.8±6.9, Time: 32.1±5.2)  
Cluster 2: 35 transactions (Amount: ₦45.3±7.4, Time: 27.8±3.9)
Anomaly Statistics

Detection Rate: 10% (industry standard: 5-15%)
Severity Levels: 5 Critical, 3 High, 2 Moderate
Primary Pattern: Amount-based anomalies (300-1200 vs 30-70 normal range)

 Limitations & Recommendations
Current Limitations

Static Approach: Doesn't adapt to evolving transaction patterns
Context Ignorance: Missing user history, account type, temporal patterns
Scale Dependency: Requires careful feature normalization
Outlier Sensitivity: Extreme values distort cluster centers

Production Recommendations

Hybrid Systems: Combine with rule-based checks and domain expertise
Feature Engineering: Add user context, temporal features, transaction sequences
Ensemble Methods: Use multiple algorithms (Isolation Forest, DBSCAN)
Continuous Learning: Regular model retraining with new data
Human-in-the-Loop: Manual review for high-risk flagged transactions

 Alternative Approaches
For production financial systems, consider:

Isolation Forest: Designed specifically for anomaly detection
DBSCAN: Density-based clustering, handles noise automatically
Autoencoders: Neural networks for complex pattern learning
Time Series Models: Handle temporal dependencies
Ensemble Methods: Combine multiple detection algorithms

 Business Impact
Immediate Value

Identified 5 critical high-value transactions requiring investigation
Clear separation of normal daily transactions from unusual large amounts
Provides risk scoring for transaction prioritization

Strategic Benefits

Foundation for scalable fraud detection system
Data-driven approach to risk management
Baseline for measuring detection performance improvements

 License
This project is licensed under the MIT License - see the LICENSE file for details.
 Author
[Beris Benjamin]

GitHub: @Beris23
LinkedIn: https://www.linkedin.com/in/beris-benjamin-0a4259228/
Email: berisrimala20@gmail.com

⭐ please Star this repository if you found it helpful!
