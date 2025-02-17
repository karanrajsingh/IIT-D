import linear_regression
import numpy as np

model = linear_regression.LinearRegressor()
path_X = './../data/Q1/linearX.csv'
path_Y = './../data/Q1/linearY.csv'
X, y = model.readData(path_X, path_Y)
arr = [0.001, 0.025, 0.1]
for i in range(len(arr)):
    X_train, y_train, X_test, y_test = model.splitTrainTest(X, y)
    param = model.fit(X_train,y_train, arr[i])
    y_train_predicted = model.predict(X_train)
    y_test_predicted = model.predict(X_test)
    train_error = model.error(y_train, y_train_predicted)
    test_error = model.error(y_test, y_test_predicted)
    print(train_error, test_error)
    #model.plot(X_train, y_train)