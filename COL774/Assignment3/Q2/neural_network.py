import numpy as np
import pandas as pd
import sys
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import time

# Preprocessing function: normalize and flatten
def pre_process_image(image):
    return (image.flatten() / 255.0)

# Read and preprocess test data
def read_images_test(path):
    path_test_image = os.path.join(path, "test")
    path_test_label = os.path.join(path, "test_labels.csv")
    X_test, y_test = read_images_class_test(path_test_image, path_test_label)
    return X_test, y_test

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
            label = df_test.loc[df_test["image"] == image, "label"].values[0]
            label_list.append(label)
        else:
            corrupted_images.append(image)

    X = np.array(array_list)
    y = np.array(label_list).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y)

    return X, y_one_hot

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
    y_one_hot = encoder.fit_transform(y_train)

    return X_train, y_one_hot

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


class NeuralNetwork:
    def __init__(self, hidden_layers, activation, mini_batch, features, classes, learning_rate, threshold = 1e-3, epochs = 300, isAdaptive = False, leaky_relu_slope = 0.01):
        print("Initializing...")
        hidden_layers.insert(0, features)
        hidden_layers.append(classes)
        self.layers = hidden_layers
        self.activation = activation
        self.W = {}
        self.b = {}
        self.mini_batch = mini_batch
        self.derivatives = {}
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.epochs = epochs
        self.isAdaptive = isAdaptive
        self.leaky_relu_slope = leaky_relu_slope


    def sigmoid(self, X):
        output = 1/(1 + np.exp(-X))
        derivative = output * (1 - output)
        return (output, derivative)
    
    def relu(self,x):
        output = np.maximum(x, 0)
        derivative = np.where(x>0, 1, 0)
        return (output, derivative)

    def leaky_relu(self, X):
        output = np.maximum(X, X*self.leaky_relu_slope)
        derivative = np.where(X>0, 1, self.leaky_relu_slope)
        return (output, derivative)
    
    def softmax(self, X):
        output = np.exp(X)/np.exp(X).sum(axis = 0)
        # print("Softmax: ", output.shape)
        return output
    
    def learning_rate_function(self, epoch):
        if self.isAdaptive:
            return self.learning_rate/np.sqrt(epoch)
        else:
            return self.learning_rate

    def cross_entropy(self, y, y_predicted):
        # print("Cross entropy: ")
        loss = -np.sum(y * np.log(y_predicted + 1e-12))/y.shape[0]
        return loss
        
    def initialize(self):
        for i in range(1, len(self.layers), 1):
            self.W["W_" + str(i)] = (1/np.sqrt(self.layers[i-1]))*np.random.randn(self.layers[i], self.layers[i-1])
            self.b["b_" + str(i)] = np.zeros(self.layers[i]).reshape([self.layers[i], 1])        

    def activation_function(self, X):
        if self.activation == "sigmoid":
            return self.sigmoid(X)
        elif self.activation == "relu":
            return self.relu(X)
        elif self.activation == "leaky_relu":
            return self.leaky_relu(X)

    def forward_prop(self, X):
        net = {}
        output = {}
        out_layer_index = len(self.layers) - 1
        output["O_0"] = X.T # as here x vector is stored as row vector
        for i in range(1, len(self.layers)):
            w = self.W["W_" + str(i)]
            b = self.b["b_" + str(i)]
            ip = output["O_" + str(i - 1)]
            # print("forward prop: ")
            # print(w.shape, ip.shape, b.shape)
            net["net_" + str(i)] = np.matmul(w, ip) + b
            # print(net["net_" + str(i)].shape)
            if i < out_layer_index:
                output["O_" + str(i)] = self.activation_function(net["net_" + str(i)])[0]            
            else:
                output["O_" + str(i)] = self.softmax(net["net_" + str(i)])
        
            # print(output["O_" + str(i)].shape)

        return output, net


    def back_prop(self, X, y, net, output):

        # print("Back prop: "

        # Output layer
        final = output["O_"+str(len(self.layers) - 1)]  # Output of last layer (softmax)
        prev = output["O_"+ str(len(self.layers) - 2)]  # Output of previous layer

        delta = final - y
        
        self.derivatives["dW_" + str(len(self.layers) - 1)] = np.dot(delta, prev.T) / X.shape[1]
        self.derivatives["db_" + str(len(self.layers) - 1)] = np.sum(delta, axis=1, keepdims=True) / X.shape[1]

 
        for i in range(len(self.layers) - 2, 0, -1):
            W_next = self.W["W_" + str(i+1)]
            delta = np.dot(W_next.T, delta)

            Z_curr = net["net_"+str(i)]
            dZ = self.activation_function(Z_curr)[1]

            delta = delta * dZ

            prev = output["O_" + str(i-1)]

            self.derivatives["dW_"+ str(i)] = np.dot(delta, prev.T) / X.shape[1]
            self.derivatives["db_"+ str(i)] = np.sum(delta, axis=1, keepdims=True) / X.shape[1]

        return self.derivatives

    def fit(self, X, y):
        print("Training...")
        examples = X.shape[0]
        self.initialize()
        print(X.shape, y.shape)
        Xy = np.concatenate([X, y], axis = 1)
        np.random.shuffle(Xy)
        batches = []
        for i in range(0, examples, self.mini_batch):
            batch = Xy[i:i + self.mini_batch]
            batches.append(batch)
        loss_str = 0
        loss_end = 0
        epoch = 0
        last = y.shape[1]
        while (abs(loss_end - loss_str) > self.threshold and epoch <= self.epochs) or epoch < 2:
            epoch_loss = 0
            for batch in batches:
                yb = batch[:, -last:]
                xb = batch[:, :-last]
                # print("batch:")
                # print(xb.shape, yb.shape)
                # print(yb)
                # print(y[:32])
                output, net = self.forward_prop(xb)
                yp = output["O_" + str(len(self.layers)-1)].T
                batch_loss = self.cross_entropy(yb, yp)
                epoch_loss += batch_loss
                self.derivatives = self.back_prop(xb.T, yb.T, net, output)
                for i in range(1, len(self.layers), 1):
                    self.W["W_" + str(i)] -= self.learning_rate_function(epoch+1) * self.derivatives["dW_" + str(i)] 
                    self.b["b_" + str(i)] -= self.learning_rate_function(epoch+1) * self.derivatives["db_" + str(i)] 
            epoch += 1
            epoch_loss = epoch_loss/examples
            loss_str = loss_end
            loss_end = epoch_loss 
            print("Epoch: {0}, Loss: {1}".format(epoch, epoch_loss))
        
        return self.W, self.b
    
    def predict(self, X):
        '''
        Predicts the output for given input X
        Args : X : Input of shape (num_examples, n_features)
        Returns : pred : Predicted output of shape (num_examples, 1)
        '''
        output = self.forward_prop(X)[0]
        prediction_array = output["O_"+str(len(self.layers)-1)].T
        prediction = prediction_array.argmax(axis = 1)
        # output_df = pd.DataFrame({'Prediction': prediction })
        # output_df.to_csv("prediction.csv")
        return prediction
    
def main():
    
    path_train = sys.argv[1]
    path_test  = sys.argv[2]
    output_folder = sys.argv[3]
    q_part = sys.argv[4].lower()


    X_trn, y_trn = read_images_train(path_train)
    X_tst, y_tst = read_images_test(path_test)

    if q_part == "b":
        hidden_layers = [[1], [5], [10], [50], [100]]
        f1_avg_train = []
        f1_avg_test = []
        prediction_list = []
        l = [1, 5, 10, 50, 100]
        for hidden in hidden_layers:
            X_train = X_trn.copy()
            y_train = y_trn.copy()
            X_test = X_tst.copy()
            y_test = y_tst.copy()
            model = NeuralNetwork(hidden_layers=hidden, activation="sigmoid", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01)
            model.fit(X_train, y_train)

            print ("Hidden layer: ", hidden)
            print("Training report: ")
            y_train = y_train.argmax(axis = 1)
            y_train_predicted = model.predict(X_train)
            report = classification_report(y_train, y_train_predicted, output_dict=True)
            f1_avg_train.append(report['macro avg']['f1-score'])
            result = classification_report(y_train, y_train_predicted)
            print(result)

            print("Testing report: ")
            y_test = y_test.argmax(axis = 1)
            y_test_predicted = model.predict(X_test)
            report = classification_report(y_test, y_test_predicted, output_dict=True)
            f1_avg_test.append(report['macro avg']['f1-score'])
            result = classification_report(y_test, y_test_predicted)
            print(result)
            prediction_list.append(y_test_predicted)

        # F1-avg v/s neurons
        plt.figure()
        plt.plot(l, f1_avg_train, label="F1 averages for train set", marker='o')
        plt.plot(l, f1_avg_test, label="F1 averages for test set", marker='o')
        plt.xlabel("Neurons")
        plt.ylabel("F1-avg")
        plt.title("(Part b) Neural Network: F1-Avg vs Number of hidden units")
        plt.legend()
        plt.grid(True)
        plt.show()  

        df_predicted = pd.DataFrame(prediction_list[-1], columns=['Predicted'])
        df_predicted.to_csv(f"{output_folder}/predicted_b.csv", index=False)


    elif q_part == "c":
        hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
        f1_avg_train = []
        f1_avg_test = []
        l = [1, 2, 3, 4]
        for hidden_layer in hidden_layers:
            flag = False
            if hidden_layer == [512, 256, 128, 64]:
                flag = True
            X_train = X_trn.copy()
            y_train = y_trn.copy()
            X_test = X_tst.copy()
            y_test = y_tst.copy()

            t1 = time.time()
            model = NeuralNetwork(hidden_layers=hidden_layer, activation="sigmoid", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01)
            model.fit(X_train, y_train)
            t2 = time.time()

            print("Training time: ", (t2-t1))

            print ("Hidden layer: ", hidden_layer)
            print("Training report: ")
            y_train = y_train.argmax(axis = 1)
            y_train_predicted = model.predict(X_train)
            report = classification_report(y_train, y_train_predicted, output_dict=True)
            f1_avg_train.append(report['macro avg']['f1-score'])
            result = classification_report(y_train, y_train_predicted)
            print(result)

            print("Testing report: ")
            y_test = y_test.argmax(axis = 1)
            y_test_predicted = model.predict(X_test)
            report = classification_report(y_test, y_test_predicted, output_dict=True)
            f1_avg_test.append(report['macro avg']['f1-score'])
            result = classification_report(y_test, y_test_predicted)
            print(result)
            if flag:
                df_predicted = pd.DataFrame(y_test_predicted, columns=['Predicted'])
                df_predicted.to_csv(f"{output_folder}/predicted_c.csv", index=False)
        # F1-avg v/s network depth
        plt.figure()
        plt.plot(l, f1_avg_train, label="F1 averages for train set", marker='o')
        plt.plot(l, f1_avg_test, label="F1 averages for test set", marker='o')
        plt.xlabel("Network Depth")
        plt.ylabel("F1-avg")
        plt.title("(Part c) Neural Network: F1-Avg vs Network Depth")
        plt.legend()
        plt.grid(True)
        plt.show() 

    elif q_part == "d":
        hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
        f1_avg_train = []
        f1_avg_test = []
        l = [1, 2, 3, 4]

        for hidden_layer in hidden_layers:
            flag = False
            if hidden_layer == [512, 256, 128, 64]:
                flag = True
            X_train = X_trn.copy()
            y_train = y_trn.copy()
            X_test = X_tst.copy()
            y_test = y_tst.copy()
            t1 = time.time()
            model = NeuralNetwork(hidden_layers=hidden_layer, activation="sigmoid", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01, isAdaptive=True)
            # model = NeuralNetwork(threshold=1e-4, hidden_layers=hidden_layer, activation="sigmoid", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01, isAdaptive=True)
            model.fit(X_train, y_train)
            t2 = time.time()

            print("Training time: ", (t2-t1))
            
            print ("Hidden layer: ", hidden_layer)
            print("Training report: ")
            y_train = y_train.argmax(axis = 1)
            y_train_predicted = model.predict(X_train)
            report = classification_report(y_train, y_train_predicted, output_dict=True)
            f1_avg_train.append(report['macro avg']['f1-score'])
            result = classification_report(y_train, y_train_predicted)
            print(result)

            print("Testing report: ")
            y_test = y_test.argmax(axis = 1)
            y_test_predicted = model.predict(X_test)
            report = classification_report(y_test, y_test_predicted, output_dict=True)
            f1_avg_test.append(report['macro avg']['f1-score'])
            result = classification_report(y_test, y_test_predicted)
            print(result)
            if flag:
                df_predicted = pd.DataFrame(y_test_predicted, columns=['Predicted'])
                df_predicted.to_csv(f"{output_folder}/predicted_d.csv", index=False)
        # F1-avg v/s network depth
        plt.figure()
        plt.plot(l, f1_avg_train, label="F1 averages for train set", marker='o')
        plt.plot(l, f1_avg_test, label="F1 averages for test set", marker='o')
        plt.xlabel("Network Depth")
        plt.ylabel("F1-avg")
        plt.title("(Part d) Neural Network: F1-Avg vs Network Depth")
        plt.legend()
        plt.grid(True)
        plt.show() 
    elif q_part == "e":
        hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
        f1_avg_train = []
        f1_avg_test = []
        l = [1, 2, 3, 4]
        for hidden_layer in hidden_layers:
            flag = False
            if hidden_layer == [512, 256, 128, 64]:
                flag = True
            X_train = X_trn.copy()
            y_train = y_trn.copy()
            X_test = X_tst.copy()
            y_test = y_tst.copy()
            model = NeuralNetwork(hidden_layers=hidden_layer, activation="relu", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01, isAdaptive=True)
            # model = NeuralNetwork(threshold=1e-4, hidden_layers=hidden_layer, activation="relu", mini_batch=32, features=X_train.shape[1], classes= 43, learning_rate=0.01, isAdaptive=True)
            model.fit(X_train, y_train)

            print ("Hidden layer: ", hidden_layer)
            print("Training report: ")
            y_train = y_train.argmax(axis = 1)
            y_train_predicted = model.predict(X_train)
            report = classification_report(y_train, y_train_predicted, output_dict=True)
            f1_avg_train.append(report['macro avg']['f1-score'])
            result = classification_report(y_train, y_train_predicted)
            print(result)

            print("Testing report: ")
            y_test = y_test.argmax(axis = 1)
            y_test_predicted = model.predict(X_test)
            report = classification_report(y_test, y_test_predicted, output_dict=True)
            f1_avg_test.append(report['macro avg']['f1-score'])
            result = classification_report(y_test, y_test_predicted)
            print(result)
            if flag:
                df_predicted = pd.DataFrame(y_test_predicted, columns=['Predicted'])
                df_predicted.to_csv(f"{output_folder}/predicted_e.csv", index=False)
        # F1-avg v/s network depth
        plt.figure()
        plt.plot(l, f1_avg_train, label="F1 averages for train set", marker='o')
        plt.plot(l, f1_avg_test, label="F1 averages for test set", marker='o')
        plt.xlabel("Network Depth")
        plt.ylabel("F1-avg")
        plt.title("(Part e) Neural Network: F1-Avg vs Network Depth")
        plt.legend()
        plt.grid(True)
        plt.show()
    elif q_part == "f":
        hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
        f1_avg_train = []
        f1_avg_test = []
        l = [1, 2, 3, 4]
        for hidden_layer in hidden_layers:
            flag = False
            if hidden_layer == [512, 256, 128, 64]:
                flag = True
            X_train = X_trn.copy()
            y_train = y_trn.copy()
            X_test = X_tst.copy()
            y_test = y_tst.copy()
            
            # Convert y_train and y_test to integer labels if they are one-hot encoded.
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_train_simple = np.argmax(y_train, axis=1)
            else:
                y_train_simple = y_train.ravel()
            if y_test.ndim > 1 and y_test.shape[1] > 1:
                y_test_simple = np.argmax(y_test, axis=1)
            else:
                y_test_simple = y_test.ravel()
            
            clf = MLPClassifier(hidden_layer_sizes=hidden_layer, activation='relu', solver='sgd', alpha=0, batch_size=32, learning_rate='invscaling', verbose = True, tol=1e-3, max_iter=300)
            clf.fit(X_train, y_train_simple)

            print ("Hidden layer: ", hidden_layer)
            print("Training report: ")
            y_train = y_train.argmax(axis = 1)
            y_train_predicted = clf.predict(X_train)
            report = classification_report(y_train, y_train_predicted, output_dict=True)
            f1_avg_train.append(report['macro avg']['f1-score'])
            result = classification_report(y_train, y_train_predicted)
            print(result)

            print("Testing report: ")
            y_test = y_test.argmax(axis = 1)
            y_test_predicted = clf.predict(X_test)
            report = classification_report(y_test_simple, y_test_predicted, output_dict=True)
            f1_avg_test.append(report['macro avg']['f1-score'])
            result = classification_report(y_test_simple, y_test_predicted)
            print(result)
            if flag:
                df_predicted = pd.DataFrame(y_test_predicted, columns=['Predicted'])
                df_predicted.to_csv(f"{output_folder}/predicted_f.csv", index=False)
        # F1-avg v/s network depth
        plt.figure()
        plt.plot(l, f1_avg_train, label="F1 averages for train set", marker='o')
        plt.plot(l, f1_avg_test, label="F1 averages for test set", marker='o')
        plt.xlabel("Network Depth")
        plt.ylabel("F1-avg")
        plt.title("(Part f) Neural Network: F1-Avg vs Network Depth")
        plt.legend()
        plt.grid(True)
        plt.show()        

if __name__ == "__main__":
    main()

