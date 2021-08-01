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

"""
X_train, X_test, y_train, y_test = train_test_split(data[1:, 0:-1], data[1:, -1], random_state=3)
#25
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

X_train = X_train / X_train.max(axis=0) * 12.5
X_test = X_test / X_test.max(axis=0) * 12.5

#ridge = Ridge().fit(X_train, y_train)
#lasso = Lasso(alpha=0.01).fit(X_train, y_train)
#ols = LinearRegression().fit(X_train, y_train)
#knn = KNeighborsRegressor(n_neighbors=31).fit(X_train, y_train)
lasso = Lasso()
ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)
#knn cross-validation for # of neighbors

knn = KNeighborsRegressor()

params = {'alpha':[0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0]}

model = GridSearchCV(ridge, params, cv=5)
model.fit(X_train,y_train)
model.best_params_
print(model.best_params_)
#print(dir(model))

print(ridge.score(X_test, y_test))
"""
"""
#print("훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train, y_train)))
a = lasso.score(X_test, y_test)
b = ridge.score(X_test, y_test)
c = ols.score(X_test, y_test)
d = knn.score(X_test, y_test)

#print("테스트 세트의 정확도 : {:.2f}".format(a))
lst.append(a)
lst2.append(b)
lst3.append(c)
lst4.append(d)

print("LASSO:",lasso.coef_)
print("RIDGE:",ridge.coef_)
print("OLS:",ols.coef_)
print("KNN:",knn.effective_metric_params_)

#print(lst)
print(lst4)

print("LST:",print(np.mean(sorted(lst, reverse = True))))
print("LST2:",print(np.mean(sorted(lst2, reverse=True))))
print("LST3:",print(np.mean(sorted(lst3, reverse=True))))
print("LST4:",print(np.mean(sorted(lst4, reverse = True))))

#plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
#plt.plot(ridge.coef_, 'o', label="Ridge alpha=0.1")
"""
