import numpy as np
import cv2
import os

"""
1. Read images for all classes.
2. Process them as: Resize to (100, 100, 3) -> centre crop -> flatten -> pixel normalization.
3. Store images as:
    var:: train/test:
                class 'dew'         : X_var_dew
                class 'fogsmog''    : X_var_dew
                :
                :
                class 'i'           : X_var_i
                :
                :
                class 'snow'        : X_var_snow
"""


def read_images(path):
    array_list_train = []
    array_list_test = []
    array_list_train_corrupted = []
    array_list_test_corrupted = []
    path_train = os.path.join(path, "train")
    path_test = os.path.join(path, "test")
    label_train = [label for label in os.listdir(path_train)]
    label_test = [label for label in os.listdir(path_test)]
    for label in label_train:
        path_train_label = os.path.join(path_train, label)
        X_train, X_train_corrupted = read_images_class(path_train_label)
        array_list_train.append(X_train)
        array_list_train_corrupted.append(X_train_corrupted)
    for label in label_test:
        path_test_label = os.path.join(path_test, label)
        X_test, X_test_corrupted = read_images_class(path_test_label)
        array_list_test.append(X_test)
        array_list_test_corrupted.append(X_test_corrupted)
    train = array_list_train
    test = array_list_test
    return array_list_train, array_list_test

def read_images_class(path):
    array_list = []
    array_list_corrupted = []
    images = [image for image in os.listdir(path)]
    for image in images:
        image_path = os.path.join(path, image)
        image_read = cv2.imread(image_path)
        if image_read is not None:
            x = pre_process_image(image_read)
            array_list.append(x)
        else:
            array_list_corrupted.append(image_path)
    X = np.array(array_list)
    return X, array_list_corrupted

def pre_process_image(image):
    update = cv2.resize(image, (100, 100))
    cropped = update[0:100, 0:100]
    flatten = cropped.flatten()
    normalized = np.multiply(flatten, 1/255)
    return normalized

'''
1. Builds inputs (X,y) for fit function for binary classfication.
2. Concatenates the X_train_Class_0 and X_train_Class_1 numpy arrays.
3. Concatenates the y_train_Class_0 and y_train_Class_1 numpy arrays.
    -Here y is -1 and 1
4. Repeats steps 2 & 3 for test sets.
5. Returns these 4 values
'''

def build_input(array_list_train, array_list_test, class_index_0 = 0, class_index_1 = 1):
    train = array_list_train
    test = array_list_test
    X_train_0 = train[class_index_0]
    X_train_1 = train[class_index_1]
    X_train = np.vstack((X_train_0, X_train_1))
    m0 = X_train_0.shape[0]
    m1 = X_train_1.shape[0]
    y_train_0 = -1*np.ones(m0)
    y_train_1 = np.ones(m1)
    y_train = np.hstack((y_train_0, y_train_1))
    X_test_0 = test[class_index_0]
    X_test_1 = test[class_index_1]
    X_test = np.vstack((X_test_0, X_test_1))
    m0 = X_test_0.shape[0]
    m1 = X_test_1.shape[0]
    y_test_0 = -1*np.ones(m0)
    y_test_1 = np.ones(m1)
    y_test = np.hstack((y_test_0, y_test_1))
    return X_train, y_train, X_test, y_test


# Pre-processing functions
def build_complete_test_input(array_list_test):
    test = array_list_test
    X = test[0]
    y = np.zeros(test[0].shape[0])
    for i in range(len(test)):
        X = np.vstack((X, test[i]))
        y = np.hstack((y, i*np.ones(test[i].shape[0])))
    return X, y

def build_complete_train_input(array_list_train):
    train = array_list_train
    X = train[0]
    y = np.zeros(train[0].shape[0])
    for i in range(len(train)):
        X = np.vstack((X, train[i]))
        y = np.hstack((y, i*np.ones(train[i].shape[0])))
    return X, y

