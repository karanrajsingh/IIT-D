import sampling_sgd
model = sampling_sgd.StochasticLinearRegressor()
model.plot(model.X_train, model.y_train)