import sampling_sgd
import numpy as np
import math
model = sampling_sgd.StochasticLinearRegressor()
N = int(1e6)
input_theta = np.array([3, 1, 2])
input_mean = np.array([3, -1])
input_sigma = np.array([2, 2])
noise_sigma = math.sqrt(2)
X_train, y_train, X_test, y_test = sampling_sgd.generate(N, input_theta, input_mean, input_sigma, noise_sigma)

param_list = model.fit(X_train, y_train, 0.001)
for i in range(len(param_list)):
    print(param_list[i][param_list[i].shape[0] - 1])
y_predicted_test_list = model.predict(X_test)
y_predicted_train_list = model.predict(X_train)
train_error_list = model.error(y_predicted_train_list, y_train)
test_error_list = model.error(y_predicted_test_list, y_test)
print(train_error_list, test_error_list)
model.plot(X_train, y_train)