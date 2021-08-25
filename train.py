#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

import os
# In[4]:

print(os.getcwd())


dataset = pd.read_csv('trainer/pima-indians-diabetes.csv')


# In[5]:


dataset.head()


# In[6]:


X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[16]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[17]:


BUCKET = 'gs://peter-test-diabetes'


# In[18]:

import joblib
joblib.dump(clf, 'model/random_forest.joblib')






