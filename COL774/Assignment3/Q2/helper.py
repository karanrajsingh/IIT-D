import cv2
import os
import numpy as np

def read_images_train(path):
    array_list_train = []
    array_list_train_corrupted = []
    path_train = os.path.join(path, "train")
    label_train = [label for label in os.listdir(path_train)]
    for label in label_train:
        path_train_label = os.path.join(path_train, label)
        X_train, X_train_corrupted = read_images_class(path_train_label)
        array_list_train.append(X_train)
        array_list_train_corrupted.append(X_train_corrupted)
    train = array_list_train
    return array_list_train

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
    return image.flatten()