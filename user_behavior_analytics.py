import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Replace 'user_behavior.csv' with the actual path to your CSV file
data = pd.read_csv('user_behavior.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Step 2: Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values (if any)
# Assuming 'location' might have missing values, fill with 'Unknown'
data['location'] = data['location'].fillna('Unknown')

# Step 3: Convert categorical features to numerical
# Use one-hot encoding for the 'location' feature
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Step 4: Standardize numerical features
scaler = StandardScaler()
data[['login_time', 'action_count']] = scaler.fit_transform(data[['login_time', 'action_count']])

# Step 5: Exploratory Data Analysis
# Plot histograms of numerical features
print("\nPlotting histograms of numerical features:")
data[['login_time', 'action_count']].hist(bins=20, figsize=(10, 5))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Plot a pairplot to visualize relationships
print("\nPlotting pairplot:")
sns.pairplot(data[['login_time', 'action_count']])
plt.suptitle('Pairplot of Numerical Features')
plt.show()

# Step 6: Apply K-means Clustering
# Define the number of clusters (e.g., 3)
print("\nApplying K-means clustering:")
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(data[['login_time', 'action_count']])

# Visualize the clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='login_time', y='action_count', hue='cluster', palette='viridis')
plt.title('User Behavior Clustering')
plt.xlabel('Login Time (Standardized)')
plt.ylabel('Action Count (Standardized)')
plt.show()

# Step 7: Identify Anomalies
# Calculate the distance of each point to its cluster center
data['distance_to_center'] = np.linalg.norm(data[['login_time', 'action_count']].values - kmeans.cluster_centers_[data['cluster']], axis=1)

# Set a threshold for identifying anomalies (e.g., 95th percentile)
threshold = data['distance_to_center'].quantile(0.95)
data['anomaly'] = data['distance_to_center'] > threshold

# Visualize anomalies
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='login_time', y='action_count', hue='anomaly', palette='coolwarm')
plt.title('Anomalies in User Behavior')
plt.xlabel('Login Time (Standardized)')
plt.ylabel('Action Count (Standardized)')
plt.show()

# Step 8: Review the Anomalies
print("\nAnomalous user behavior detected:")
anomalies = data[data['anomaly'] == True]
print(anomalies)

# Save the results to a new CSV file
anomalies.to_csv('anomalous_user_behavior.csv', index=False)
print("\nAnomalies saved to 'anomalous_user_behavior.csv'.")
