import helper
import numpy as np

path_train = "../data/Q2"

array_list_train = helper.read_images_train(path_train)

print(len(array_list_train))
for i in range(len(array_list_train)):
    print(len(array_list_train[i]))
    # for j in range(len(array_list_train[i])):
    #     print(array_list_train[i][j].shape)

