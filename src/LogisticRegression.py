
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
    
    def __init__(self, learning_rate=0.001,num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        num_samples = len(X)
        num_features = len(X[0])

        # initialize weights and bias to 0
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_iterations):

            # pred = sum wi * xi + bias
            linear_pred = np.dot(X, self.weights) + self.bias

            #apply sigmoid function
            y_pred = sigmoid(linear_pred)

            # calculate the gradients for loss
            weight_grad = (1/num_samples) * np.dot(X.T, (y_pred-y))
            bias_grad = (1/num_samples) * np.sum(y_pred-y)

            # adjust each of the weights
            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad

    def predict(self, X):
        #print(self.weights)
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<= 0.5 else 1 for y in y_pred]
        return y_pred, class_pred






