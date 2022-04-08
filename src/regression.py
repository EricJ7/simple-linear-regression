import numpy as np

class linearRegression:
    def __init__(self, iterations, learningRate):
        self.iterations = iterations
        self.learningRate = learningRate


    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        # initiate weight and bias value to 0
        self.weight = 0
        self.bias = 0

        # update weights for every iteration
        for i in range(self.iterations):
            self.updateWeights()

        return

    # compute gradient for bias
    # (partial der. of mse with respect to b)
    def biasGradient(self):
        N = len(self.X)
        diff = 0

        for i in range(N):
            x = self.X[i]
            y = self.Y[i]
            diff += (y - ((self.weight * x) + self.bias))

        biasGradient = -(2/N) * diff

        return biasGradient

    # computing weight gradient
    # (partial der. of mse with respect to w)
    def weightGradient(self):
        N = len(self.X)
        diff = 0

        for i in range(N):
            x = self.X[i]
            y = self.Y[i]
            diff += x * ((y - ((self.weight * x) + self.bias)))

        weightGradient = -(2/N) * diff

        return weightGradient

    # updating weight and bias value based on gradients
    def updateWeights(self):

        weightGradient = self.weightGradient()
        biasGradient = self.biasGradient()

        # update weight and bias value
        # wnew = wold - (lr * dw)
        # bnew = bold - (lr * db)
        self.weight = self.weight - (self.learningRate * weightGradient)
        self.bias = self.bias - (self.learningRate * biasGradient)


    # single variable prediction
    def predict(self, x):

        return self.weight * x + self.bias
