import helper
import numpy as np
import neural_network

path = "../data/Q2"
X_train, y_train = helper.read_images_train(path)
X_test, y_test = helper.read_images_test(path)
# print(y_train)
print("X_train: ")
print(X_train.shape, y_train.shape)
print("X_test: ")
print(X_test.shape, y_test.shape)
# print(y_test)
# # print(array_list_test)

model = neural_network.NeuralNetwork(hidden_layers=[100], activation="sigmoid", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01)
model.fit(X_train, y_train)
