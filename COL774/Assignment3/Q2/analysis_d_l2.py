import Q2.helper_old as helper_old
import numpy as np
import neural_network
from sklearn.metrics import classification_report

path = "../data/Q2"
X_train, y_train = helper_old.read_images_train(path)
X_test, y_test = helper_old.read_images_test(path)
# print(y_train)
print("X_train: ")
print(X_train.shape, y_train.shape)
print("X_test: ")
print(X_test.shape, y_test.shape)
# print(y_test)
# # print(array_list_test)

model = neural_network.NeuralNetwork(hidden_layers=[512, 256], activation="sigmoid", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01, isAdaptive=True)
model.fit(X_train, y_train)
y_test = y_test.argmax(axis = 1)
y_test_predicted = model.predict(X_test)
result = classification_report(y_test, y_test_predicted)
print("Shapes: ",y_test_predicted.shape, y_test.shape)
print(result)

# y_train = y_train.argmax(axis = 1)
# y_train_predicted = model.predict(X_train)
# print("Shapes: ",y_train_predicted.shape, y_train.shape)
# result = classification_report(y_train, y_train_predicted)
# print(result)
