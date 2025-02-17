import logistic_regression
import numpy as np
model = logistic_regression.LogisticRegressor()
X, y = model.read(path_X ='./../data/Q3/logisticX.csv', path_Y = './../data/Q3/logisticY.csv')
theta = model.fit(X,y)
#param = theta[len(theta)-1]
model.plot(X, y)