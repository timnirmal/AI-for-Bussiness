# https://medium.com/codex/customer-segmentation-using-k-means-clustering-algorithm-4c125d29d11d

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

################################################## 1. Exploring the data

df = pd.read_csv("Mall_Customers.csv")
# Printing first 5 records of the data frame
print(df.head())
"""
# Checking for the count of rows and columns
print(df.shape)
# Looking at the datatypes on the columns
print(df.info())
# Checking for null values
print(df.isnull().sum())

# Plotting box plots to check for outliers
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
sns.boxplot(x='Age',data=df)
plt.subplot(3,1,2)
sns.boxplot(x='Annual Income (k$)',data=df)
plt.subplot(3,1,3)
sns.boxplot(x='Spending Score (1-100)',data=df)
plt.show()

# Statistical information
print(df.describe())

# Male vs Female vistors to the mall
sns.set_theme(style="whitegrid")
ax1=sns.countplot(x='Gender',data=df)
plt.show()

# PDF of Age
ax3 = sns.distplot(df['Age'])
# plot
plt.show()

"""

################################################## 2.Selecting Annual Income and Spending Score for further analysis

X = df.iloc[:, 3:5].values
print(X[0:5])

################################################## 3. Identifying the number of clusters using elbow graph

# Computing WSS scored for k values 1-10
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow plot
sns.set()
plt.plot(range(1, 11), wcss)
plt.title("Elbow Graph")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
# Elobow plot shows that the elbow is at k=5


# WCSS
# https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce
"""
Elbow method is one of the most popular methods to select the optimum value of k for the model.

It works by calculating the Within-Cluster-Sum of Squared Errors ( WSS ) for different values of k and choose the value of k for which the WSS diminishes the most.

Lets breakdown WSS:

The squared error for each point is the square of the distance between the point and the predicted cluster center.
Any distance metric such as the Euclidean distance or Hamming distance can be used.
The WSS score is the sum of these squared errors for all data points.
The point where there is a significant drop in WSS score is selected to be the value of k.
"""

################################################## 4. Applying K-Means Clustering Algorithm (Training the model)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=7)
Y = kmeans.fit_predict(X)
print(Y)

################################################## 5. Visualizing the clusters
# Scatter plot to visualize the different cluster in the dataset

plt.figure(figsize=(10, 10))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='blue', label='cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='red', label='cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='yellow', label='cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='cyan', label='cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label="Centroids")

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
