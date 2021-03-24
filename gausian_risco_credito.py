# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:54:55 2021

@author: bielj
"""

import pandas as pd

df = pd.read_csv("risco_credito.csv")
x = df.drop("c#risco", axis = 1)
y = df["c#risco"]

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
x.iloc[:, 0] = label.fit_transform(x.iloc[:, 0])
x.iloc[:, 1] = label.fit_transform(x.iloc[:, 1])
x.iloc[:, 2] = label.fit_transform(x.iloc[:, 2])
x.iloc[:, 3] = label.fit_transform(x.iloc[:, 2])

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x, y)
result = classifier.predict([[0,0,1,2]])