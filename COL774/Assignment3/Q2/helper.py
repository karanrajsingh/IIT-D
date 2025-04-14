import cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Preprocessing function: normalize and flatten
def pre_process_image(image):
    return (image.flatten() / 255.0)

# Read and preprocess test data
def read_images_test(path):
    path_test_image = os.path.join(path, "test")
    path_test_label = os.path.join(path, "test_labels.csv")
    X_test, y_test = read_images_class_test(path_test_image, path_test_label)
    return X_test, y_test

# Helper for test images: match images with labels and apply preprocessing
def read_images_class_test(image_path, label_path):
    df_test = pd.read_csv(label_path)
    array_list = []
    label_list = []
    corrupted_images = []

    for image in os.listdir(image_path):
        image_full_path = os.path.join(image_path, image)
        img = cv2.imread(image_full_path)
        if img is not None:
            x = pre_process_image(img)
            array_list.append(x)
            try:
                label = df_test.loc[df_test["image"] == image, "label"].values[0]
                label_list.append(label)
            except IndexError:
                print(f"Warning: No label found for image {image}")
        else:
            corrupted_images.append(image)

    X = np.array(array_list)
    y = np.array(label_list).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded

# Read and preprocess training images
def read_images_train(path):
    X_train = []
    y_train = []
    corrupted_images_all = []

    path_train = os.path.join(path, "train")
    labels = sorted(os.listdir(path_train), key=lambda x: int(x))  # ensure label order is consistent

    for label in labels:
        label_path = os.path.join(path_train, label)
        X_label, corrupted = read_images_class(label_path)
        y_label = int(label) * np.ones(X_label.shape[0])
        X_train.append(X_label)
        y_train.append(y_label)
        corrupted_images_all.extend(corrupted)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y_train)

    return X_train, y_encoded

# Helper for reading images from a single class
def read_images_class(path):
    array_list = []
    corrupted_images = []

    for image in os.listdir(path):
        image_path = os.path.join(path, image)
        img = cv2.imread(image_path)
        if img is not None:
            array_list.append(pre_process_image(img))
        else:
            corrupted_images.append(image_path)

    return np.array(array_list), corrupted_images
