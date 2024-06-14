# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, roc_auc_score

import pickle

import os
os.getcwd()

os.chdir('C:\\Users\\Lenovo\\Downloads')
loaded_model = pickle.load(open('trained_model.sav','rb'))

input_data = (5,0.2,0,3,0.39,0,19)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The trade has more than 85% chances of hitting the target')
else:
  print('The trade has more than 85% chances of hitting the stoploss')
