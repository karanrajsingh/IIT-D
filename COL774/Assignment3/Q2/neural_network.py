import numpy as np
import pandas as pd

# path = "../data/Q2"
# X_train, y_train = helper.read_images_train(path)
# X_test, y_test = helper.read_images_test(path)
# # print(y_train)
# # print(X_train.shape, y_train.shape)
# # print(X_test.shape, y_test.shape)
# # print(y_test)
# # # print(array_list_test)

class NeuralNetwork:
    def __init__(self, hidden_layers, activation, mini_batch, features, classes, learning_rate, threshold = 1e-3, epochs = 70, isAdaptive = False, leaky_relu_slope = 0.01):
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
        derivative = np.where(X>0, 1, self.leakyRelu_slope)
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
            # self.W["W_" + str(i)] = np.random.randn(self.layers[i], self.layers[i - 1]) / np.sqrt(self.layers[i-1])
            # self.b["b_" + str(i)] = np.zeros((self.layers[i], 1))
            
            


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
    
    # def back_prop(self, X, y, net, output):
        
    #     num = X.shape[1]

    #     out = output["O_" + str(len(self.layers) - 1)]

    #     print("Back prop: ", y.shape, X.shape, out.shape, output["O_" + str(len(self.layers) - 2)].shape)

    #     delta = out - y
    #     self.derivatives["dW_" + str(len(self.layers) - 1)] = np.dot(delta, output["O_" + str(len(self.layers) - 2)].T)/num
    #     self.derivatives["db_" + str(len(self.layers) - 1)] = np.sum(delta, keepdims=True, axis=1) /num

    #     for i in range(len(self.layers) - 2, 0, -1):
    #         print(self.W["W_" + str(i+1)].T.shape, delta.shape)
    #         delta = self.W["W_" + str(i+1)].T.dot(delta)
    #         print(delta.shape, net["net_" + str(i)].shape)
    #         delta = np.multiply(delta, self.activation_function(net["net_" + str(i)]))
    #         print(delta.shape, output["O_" + str(i-1)].T.shape)
    #         self.derivatives["dW_" + str(i)] = np.dot(delta, output["O_" + str(i-1)].T)/num
    #         self.derivatives["db_" + str(i)] = np.sum(delta, keepdims=True, axis=1)/num
    #         print("param and diff param: ", self.W["W_" + str(i)].shape, self.derivatives["dW_" + str(i)].shape)
    #     return self.derivatives


    def back_prop(self, X, y, net, output):

        # print("Back prop: ")

        m = X.shape[1]  # number of samples

        # Last layer (softmax + cross-entropy)
        L = len(self.layers) - 1
        A_final = output[f"O_{L}"]  # Output of last layer (softmax)
        A_prev = output[f"O_{L - 1}"]  # Output of previous layer

        delta = A_final - y  # (num_classes, batch_size)
        
        self.derivatives[f"dW_{L}"] = np.dot(delta, A_prev.T) / m
        self.derivatives[f"db_{L}"] = np.sum(delta, axis=1, keepdims=True) / m

        # Loop over hidden layers in reverse
        for i in range(L - 1, 0, -1):
            W_next = self.W[f"W_{i+1}"]
            delta = np.dot(W_next.T, delta)

            Z_curr = net[f"net_{i}"]
            dZ = self.activation_function(Z_curr)[1]  # Use derivative, not activation itself

            delta = delta * dZ  # Element-wise multiplication

            A_prev = output[f"O_{i-1}"]

            self.derivatives[f"dW_{i}"] = np.dot(delta, A_prev.T) / m
            self.derivatives[f"db_{i}"] = np.sum(delta, axis=1, keepdims=True) / m

        return self.derivatives

    def fit(self, X, y):
        print("Training...")
        examples = X.shape[0]
        self.initialize()
        Xy = np.concatenate([X, y], axis = 1)
        np.random.shuffle(Xy)
        batches = []
        for i in range(0, examples):
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
        prediction_array = self.forward_prop(X)
        prediction = prediction_array.argmax(axis = 0)
        output_df = pd.DataFrame({'Prediction': prediction })
        output_df.to_csv("prediction")
        return prediction