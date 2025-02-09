#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("final_substructure_dataset_ul94.csv")
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target variable (classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=41)

# Data scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test = scaler.transform(X_test)        # Transform on test data

# Load additional dataset

dataset_generated_structures = pd.read_csv("ourresult1.csv")

X_generated_structures_scaler = scaler.transform(dataset_generated_structures)  # Apply the same scaling as training data

# Feature selection using Ridge Classifier
estimator = RidgeClassifier(alpha=1)
selector = RFE(estimator, n_features_to_select=85)
selector.fit(X_train, y_train)
X_train_selector = selector.transform(X_train)
X_test_selector = selector.transform(X_test)
X_generated_structures_selector = selector.transform(X_generated_structures_scaler)

# Output selected features
selected_features = selector.get_support(indices=True)
logf = open("logfile.log", "a+")
print(f"Selected feature indices: {selected_features}", file=logf, flush=True)

# Classification model
classifier = GradientBoostingClassifier(n_estimators=6500, max_depth=2, learning_rate=0.01, random_state=7)

classifier.fit(X_train_selector, y_train)

# Model Performance Evaluation
y_train_predict = classifier.predict(X_train_selector)
y_test_predict = classifier.predict(X_test_selector)

accuracy_train = accuracy_score(y_train, y_train_predict)
accuracy_test = accuracy_score(y_test, y_test_predict)
precision = precision_score(y_test, y_test_predict, average='weighted')
recall = recall_score(y_test, y_test_predict, average='weighted')
f1 = f1_score(y_test, y_test_predict, average='weighted')

# Log metrics
print(f"Train Accuracy: {accuracy_train:.3f}", file=logf, flush=True)
print(f"Test Accuracy: {accuracy_test:.3f}", file=logf, flush=True)
print(f"Test Precision: {precision:.3f}", file=logf, flush=True)
print(f"Test Recall: {recall:.3f}", file=logf, flush=True)
print(f"Test F1 Score: {f1:.3f}", file=logf, flush=True)



y_generated_structures_predict = classifier.predict(X_generated_structures_selector)  


# In[ ]:




