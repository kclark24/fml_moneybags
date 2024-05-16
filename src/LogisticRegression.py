
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
        # print(X)
        num_samples = len(X)
        num_features = len(X[0])
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)

            dw = (1/num_samples) * np.dot(X.T, (y_pred-y))
            db = (1/num_samples) * np.sum(y_pred-y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        # TODO: probably change this to return normal output becasue we want certainty
        #class_pred = y_pred
        class_pred = [0 if y<= 0.5 else 1 for y in y_pred]
        return y_pred, class_pred






