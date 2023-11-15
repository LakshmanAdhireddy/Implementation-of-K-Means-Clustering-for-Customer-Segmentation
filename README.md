# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 :
Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

STEP 2 :
Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

STEP 3 :
Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

STEP 4 :
Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

STEP 5 :
Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

STEP 6 :
Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

STEP 7 :
Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
### Developed by: Lakshman
### RegisterNumber:  212222240001
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```

## Output:

### data.head() :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707265/45463e6a-4fa2-48e6-9d7d-ca9bb4ec94e5)

### data.info() :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707265/7d189eee-1283-431d-a831-5ebade190a19)

### Null Values :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707265/f2ca7377-d2bc-4a5b-8d83-69fe9a9d029a)

### Elbow Graph :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707265/f36291f2-3696-4349-995d-7cf0b2dede42)

### K-Means Cluster Formation :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707265/8f9e76f6-3355-475b-b292-7ec50efb466c)

### Predicted Value :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707265/76e739c9-ba7b-4593-9069-fee1c8996e0c)

### Final Graph :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118707265/4455cde1-c91d-4d82-9e8e-9f2396ad1a1e)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
