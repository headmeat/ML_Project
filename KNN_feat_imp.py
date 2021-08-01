from numpy import mean
from numpy import std
from numpy import absolute
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from IPython.display import display
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


"""
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
model.best_params_
""" 

# load the dataset
df = read_csv("C:/Users/PC/Desktop/Data1.csv")

df["Temperature (K)"] = pd.to_numeric(df["Temperature (K)"], downcast="float")
df["Pressure (psi)"] = pd.to_numeric(df["Pressure (psi)"], downcast="float")
df["Thermal Conductivity (W/(m?K)"] = pd.to_numeric(df["Thermal Conductivity (W/(m?K)"], downcast="float")
df["Sonic Velocity (m/s)"] = pd.to_numeric(df["Sonic Velocity (m/s)"], downcast="float")
df["WI (MJ/Nm�)"] = pd.to_numeric(df["WI (MJ/Nm�)"], downcast="float")

dep_var = "WI (MJ/Nm�)"
cond = np.random.rand(len(df))>.2
train = np.where(cond)[0]
valid = np.where(~cond)[0]

train_df = df.iloc[train]
valid_df = df.iloc[valid]

train_y = train_df["WI (MJ/Nm�)"]
train_xs = train_df.drop(["WI (MJ/Nm�)"], axis = 1)

valid_y = valid_df["WI (MJ/Nm�)"]
valid_xs = valid_df.drop(["WI (MJ/Nm�)"], axis = 1)

m = KNeighborsRegressor()

m = m.fit(train_xs, train_y)

std_score = m.score(valid_xs, valid_y)
#print(std_score)

data = {"Temperature (K)":[0], "Pressure (psi)":[0], "Thermal Conductivity (W/(m?K)":[0], "Sonic Velocity (m/s)":[0], "WI (MJ/Nm�)":[0]}
feat_imp = pd.DataFrame(data)
#print(feat_imp.head())

valid_xs.head()

valid_SL = valid_xs.copy()
valid_SL["Temperature (K)"] = np.random.permutation(valid_SL["Temperature (K)"])
#print(valid_SL.head())

#print(m.score(valid_SL, valid_y))

feat_imp["Temperature (K)"] = std_score - m.score(valid_SL, valid_y)
#print(feat_imp.head())

valid_SW = valid_xs.copy()
valid_SW["Pressure (psi)"] = np.random.permutation(valid_SW["Pressure (psi)"])
#print(valid_SW.head())
#print(m.score(valid_SW, valid_y))

feat_imp["Pressure (psi)"] = std_score - m.score(valid_SW, valid_y)

valid_PL = valid_xs.copy()
valid_PL["Thermal Conductivity (W/(m?K)"] = np.random.permutation(valid_PL["Thermal Conductivity (W/(m?K)"])
#print(m.score(valid_PL, valid_y))
feat_imp["Thermal Conductivity (W/(m?K)"] = std_score - m.score(valid_PL, valid_y)

valid_PW = valid_xs.copy()
valid_PW["Sonic Velocity (m/s)"] = np.random.permutation(valid_PW["Sonic Velocity (m/s)"])
feat_imp["Sonic Velocity (m/s)"] = std_score - m.score(valid_PW, valid_y)

print(feat_imp["Temperature (K)"], feat_imp["Pressure (psi)"], feat_imp["Thermal Conductivity (W/(m?K)"],
      feat_imp["Sonic Velocity (m/s)"])
