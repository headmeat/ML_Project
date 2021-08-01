from numpy import mean
from numpy import std
from numpy import absolute
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

"""
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
model.best_params_
"""

# load the dataset
dataframe = read_csv("C:/Users/PC/Desktop/Data1.csv", header=None)
data = dataframe.values

lst = []
lst2 = []
lst3 = []
lst4 = []
lst5= []
lst6=[]
lst7=[]
lst8=[]

for i in range(20, 40):
    X_train, X_test, y_train, y_test = train_test_split(data[1:, 0:-1], data[1:, -1], random_state=i)
    #25
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    
    X_train = X_train / X_train.max(axis=0) * 12.5
    X_test = X_test / X_test.max(axis=0) * 12.5
    
    ridge = Ridge().fit(X_train, y_train)
    lasso = Lasso(alpha=0.01).fit(X_train, y_train)
    ols = LinearRegression().fit(X_train, y_train)
    knn = KNeighborsRegressor(n_neighbors=31).fit(X_train, y_train)
    
    #knn cross-validation for # of neighbors
    """
    knn = KNeighborsRegressor()
    
    params = {'n_neighbors':[5, 10, 15, 20, 25, 31]}
    
    model = GridSearchCV(knn, params, cv=5)
    model.fit(X_train,y_train)
    model.best_params_
    print(model.best_params_)
    """
    
    #plotting score graph for knn
    """
    scores = [x[1] for x in model.grid_scores_]
    scores = np.array(scores).reshape(len(Cs), len(Gammas))
    
    for ind, i in enumerate(Cs):
        plt.plot(Gammas, scores[ind], label='C: ' + str(i))
        
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Mean score')
    plt.show()
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

#print("사용한 특성의 수 : {}".format(np.sum(lasso.coef_ != 0)))

#print("사용한 max_iter : {}".format(lasso.n_iter_))
"""
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
lst3.append(lasso001.score(X_test, y_test))

print("훈련 세트의 정확도 : {:.2f}".format(lasso001.score(X_train, y_train)))

print("테스트 세트의 정확도 : {:.2f}".format(lasso001.score(X_test, y_test)))

print("사용한 특성의 수 : {}".format(np.sum(lasso001.coef_ != 0)))

print("사용한 max_iter : {}".format(lasso001.n_iter_))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
lst4.append(lasso00001.score(X_test, y_test))

print("훈련 세트의 정확도 : {:.2f}".format(lasso00001.score(X_train, y_train)))

print("테스트 세트의 정확도 : {:.2f}".format(lasso00001.score(X_test, y_test)))

print("사용한 특성의 수 : {}".format(np.sum(lasso00001.coef_ != 0)))

print("사용한 max_iter : {}".format(lasso00001.n_iter_))
"""

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
X, y = data[1:, 1:-1], data[1:, -1]
# define model
model = Lasso(alpha=0.7)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
"""
