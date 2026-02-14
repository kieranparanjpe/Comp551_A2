import numpy
import numpy as np
from dataset import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def logistic(z):
    return 1 / (1 + np.exp(-z.clip(-500, 500))) # clip for numerical stability

class LogisticRegression:

    def __init__(self, dataset : Dataset, lambda_ = 0, batch_size=1, learning_rate=1, epochs=50):
        self.rng = np.random.default_rng(2026)
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = self.rng.random(dataset.X.shape[1]).reshape(-1, 1)
        self.dataset= dataset
        self.trainX = self.dataset.trainX
        self.trainY = self.dataset.trainY
        self.validationX = self.dataset.trainX
        self.validationY = self.dataset.trainY

    def batched_x(self, index = 0):
        return self.trainX[index : index + self.batch_size]

    def batched_y(self, index = 0):
        return self.trainY[index : index + self.batch_size]

    def forward_batched(self, index=0):
        return self.forward(self.batched_x(index))

    def forward(self, X):
        return logistic(self.logit(X))

    def logit(self, X):
        return X @ self.weights

    def gradient(self, index=0):
        grad = (1 / self.batched_x(index).shape[0]) * self.batched_x(index).transpose() @ (self.forward_batched(index) -
                                                                            self.batched_y(index))
        decay = self.lambda_ * self.weights # using 1/2 lambda in loss function
        decay[0] = 0
        grad += decay
        return grad

    def update_weights(self, index=0):
        self.weights -= self.learning_rate * self.gradient(index)

    def ce_loss(self, X, y):
        z = self.logit(X)
        loss = (y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)).mean()
        loss += self.lambda_ / 2 * (self.weights[1:].T @ self.weights[1:])
        return loss.item()

    def accuracy(self, X, y):
        return np.mean((self.forward(X) >= 0.5).astype(int) == y)


    def train_(self):
        # epoch, train error, test error
        stats = []
        self.rng = np.random.default_rng(2026)
        self.weights = self.rng.random(self.dataset.X.shape[1]).reshape(-1, 1)

        for epoch in range(self.epochs):
            perm = self.rng.permutation(self.trainX.shape[0])
            self.trainX = self.trainX[perm]
            self.trainY = self.trainY[perm]
            for batch_index in range(0, self.trainX.shape[0], self.batch_size):
                self.update_weights(batch_index)
            stats.append(
                (epoch,
                 self.ce_loss(self.trainX, self.trainY),
                 self.ce_loss(self.validationX, self.validationY),
                 self.ce_loss(self.dataset.testX, self.dataset.testY),
                 self.accuracy(self.trainX, self.trainY),
                 self.accuracy(self.validationX, self.validationY),
                 self.accuracy(self.dataset.testX, self.dataset.testY)
                )
            )
        return np.array(stats)

    def train(self):
        self.trainX = self.dataset.trainX
        self.trainY = self.dataset.trainY
        self.validationX = self.dataset.trainX
        self.validationY = self.dataset.trainY
        return self.train_()

    def train_cross_validation(self, L=3):
        def train_single_fold(l=0):
            n = self.dataset.trainX.shape[0]
            fold_size = (n + L - 1) // L
            fold_begin, fold_end = l*fold_size, min((l + 1) * fold_size, n)
            self.trainX = np.concatenate((self.dataset.trainX[0:fold_begin], self.dataset.trainX[fold_end:]))
            self.trainY = np.concatenate((self.dataset.trainY[0:fold_begin], self.dataset.trainY[fold_end:]))
            self.validationX = self.dataset.trainX[fold_begin:fold_end]
            self.validationY = self.dataset.trainY[fold_begin:fold_end]
            return self.train_()

        fold_stats = []

        for fold_index in range(L):
            fold_stats.append(train_single_fold(fold_index))
        return np.array(fold_stats)


if __name__ == '__main__':
    d = Dataset()
    regression = LogisticRegression(d)
    stats = regression.train()

