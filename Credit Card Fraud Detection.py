#!/usr/bin/env python
# coding: utf-8

# In[15]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import sklearn
import scipy


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


# Load dataset
data = pd.read_csv("creditcard.csv")


# In[18]:


print(data.columns)
print(data.shape)


# In[19]:


print(data.describe())


# In[20]:


# Sample the data as the data is too large
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)


# In[21]:


# Histogram
data.hist(figsize = (20, 20))
plt.show()


# In[22]:


# Determine number of fraud cases
Fraud = data[data["Class"] == 1]
Valid = data[data["Class"] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))


# In[24]:


# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[25]:


# Get all columns from dataset
columns = data.columns.tolist()

# Filter columns to remove unwanted data
columns = [c for c in columns if c not in ["Class"]]

# Store variable to predict
target = "Class"

X = data[columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)


# In[26]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Define a random state
state = 1

# Define outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X), contamination=outlier_fraction, random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
}


# In[28]:


# Fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fit the data and tags
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # Reshape the prediction values
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print("{}: {}".format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:




