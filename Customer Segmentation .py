#!/usr/bin/env python
# coding: utf-8

# In[2]:
# Import necessary libraries
ip install plotly cufflinks char_studio
import cufflinks as cf 
import chart_studio.plotly as py
import plotly.express  as px
import plotly. graph_objects as go
import plotly. fi


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Ensure inline plotting
get_ipython().run_line_magic('matplotlib', 'inline')




# In[3]:


# Load your dataset
data = pd.read_csv('Customer Segmentatiom.csv')

# Display the first few rows of the dataset
data.head()


# In[9]:


# Select relevant columns for normalization
features = ['Age', 'Annual Income ($) ', 'Spending Score (1-100)', 'Work Experience', 'Family Size']

# Normalize the data
data_normalized = (data[features] - data[features].mean()) / data[features].std()

# Display the normalized data
data_normalized.head()


# In[10]:


# Display the column names to check for exact matches
print(data.columns)


# In[11]:


features = ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']


# In[12]:


# Normalize the data
data_normalized = (data[features] - data[features].mean()) / data[features].std()

# Display the normalized data
data_normalized.head()


# In[13]:


# Calculate WCSS for different values of k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_normalized)
    wcss.append(kmeans.inertia_)


# In[14]:


plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[15]:


optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_normalized)

# Display the first few rows of the dataset with the cluster assignments
data.head()


# In[16]:


# Visualize the clusters in a 2D plot (for example using Age and Annual Income ($))
plt.scatter(data['Age'], data['Annual Income ($)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Annual Income ($)')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()


# In[17]:


# Selecting additional features for clustering
features = ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']


# In[18]:


print(data_normalized.head())


# In[19]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_normalized)
    wcss.append(kmeans.inertia_)


# In[20]:


plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[21]:


optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_normalized)


# In[22]:


pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_normalized)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()


# In[23]:


plt.scatter(data['Age'], data['Annual Income ($)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Annual Income ($)')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()


# In[24]:


# Group data by cluster
clustered_data = data.groupby('Cluster').mean()

# Display summary statistics for each cluster
print(clustered_data)


# In[25]:


# Create bar charts for each feature
features_to_plot = ['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    clustered_data[feature].plot(kind='bar')
    plt.title(f'Average {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'Average {feature}')
    plt.show()


# In[26]:


# Create pie charts for categorical data if needed
# Example: Pie chart for the distribution of clusters

cluster_counts = data['Cluster'].value_counts()

plt.figure(figsize=(8, 8))
cluster_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='viridis')
plt.title('Cluster Distribution')
plt.ylabel('')  # Hides the y-label to make the plot look cleaner
plt.show()


# In[ ]:




