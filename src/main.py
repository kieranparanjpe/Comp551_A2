import numpy as np

from logistic_regression_sgd import LogisticRegression
from dataset import Dataset
import plots
from tqdm import tqdm

if __name__ == '__main__':
    d = Dataset()
    batch_sizes = [x for x in range(8, 64, 8)]
    learning_rates = [10**-x for x in range(6)]
    epochs = [x for x in range(50, 200, 25)]
    regularizations = [10**-x for x in range(6)]

    means_of_each = []

    for bs in tqdm(batch_sizes):
        for lr in learning_rates:
            for e in epochs:
                regression = LogisticRegression(d, epochs=e, lambda_=0, learning_rate=lr, batch_size=bs)
                stats = regression.train_cross_validation(3)
                # for s in stats:
                #     plots.display_epoch_train_test(s[:,0:4], "plot 1 CE", True)
                #     plots.display_epoch_train_test(np.concat((s[:,0].reshape(-1, 1), s[:,4:]), 1), "plot 1 A", True)

                means_of_each.append(( (stats[:, -1, 2].mean(), stats[:, -1, 2].std()), (bs, lr, e) ))
                 # gives mean of validation error over each fold

    print(sorted(means_of_each)[:10])