import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.linear_model import LogisticRegressionCV as SkLogisticRegressionCV
from logistic_regression_sgd import LogisticRegression
from dataset import Dataset
import plots
from tqdm import tqdm

from src.plots import plot_sparsity, plot_reg_path


def task1(d):
    batch_sizes = [1, 16, 64]
    learning_rates = [10**-x for x in range(0,5)]
    epochs = 200 # only doing 200 epochs because otherwise we generate many plots. Also, looking at graph of 200
    # epochs we can see performance after <200 epochs.
    regularizations = [0, 10**-3]

    for bs in tqdm(batch_sizes):
        for lr in learning_rates:
            for r in regularizations:
                regression = LogisticRegression(d, epochs=epochs, lambda_=r, learning_rate=lr, batch_size=bs)
                stats = regression.train()
                plots.display_epoch_train_test(stats[:, 0], stats[:, 1:4],
               f"Task 1 (BS={bs}, LR={lr}, Lambda={r}) CE Loss",False)
                plots.display_epoch_train_test(stats[:, 0], stats[:, 4:],
               f"Task 1 (BS={bs}, LR={lr}, Lambda={r}) Accuracy", False,"Accuracy")

def print_task2(arr):
    print("---------------------------------------")
    print("Task 2:\n")
    print("Top 10 best hyperparameter configurations by avg CE loss:")
    sorted_by_ce = arr[arr[:, 0].argsort()]
    print(sorted_by_ce[0:10])
    print("Top 10 best hyperparameter configurations by avg accuracy:")
    sorted_by_acc = arr[arr[:, 2].argsort()[::-1]]
    print(sorted_by_acc[0:10])

    # best model is 8.00000,0.10000,125.00000,0.10000
    # because if you inspect the best acc, std deviation includes the accuracy of the best ce, but best ce std does
    # not include ce of best acc, so we take the best ce, which has params above.

def task2_best_model(d):
    regression = LogisticRegression(d, epochs=125, lambda_=0.1, learning_rate=0.1, batch_size=8)
    stats = regression.train()
    plots.display_epoch_train_test(stats[:, 0], stats[:, 1:4],
                                   f"Task 2 Best model CE Loss", False)
    plots.display_epoch_train_test(stats[:, 0], stats[:, 4:],
                                   f"Task 2 Best model Accuracy", False, "Accuracy")

def task2(d):
    batch_sizes = [x for x in range(8, 64, 8)]
    learning_rates = [10**-x for x in range(1,5)]
    epochs = [x for x in range(25, 200, 25)]
    regularizations = [0] + [10**-x for x in range(3)]

    means_of_each = []
    for bs in tqdm(batch_sizes):
        for lr in tqdm(learning_rates):
            for e in epochs:
                for r in regularizations:
                    regression = LogisticRegression(d, epochs=e, lambda_=r, learning_rate=lr, batch_size=bs)
                    stats = regression.train_cross_validation(6)
                    # for s in stats:
                    #     plots.display_epoch_train_test(s[:,0:4], "plot 1 CE", True)
                    #     plots.display_epoch_train_test(np.concat((s[:,0].reshape(-1, 1), s[:,4:]), 1), "plot 1 A", True)
                    # gives per each fold (mean ce loss, std ce loss, mean acc, std acc, bs, lr, e, r)
                    means_of_each.append([stats[:, -1, 2].mean(), stats[:, -1, 2].std(),
                                          stats[:, -1, 5].mean(), stats[:, -1, 5].std(),
                                          bs, lr, e, r])

    arr = np.array(means_of_each)
    np.save("stats_task2.npy", arr)
    print_task2(arr)

def task3(d):
    regularizations = [0] + [10 ** -x for x in range(8)]
    means_of_each = []
    for r in regularizations:
        regression = LogisticRegression(d, epochs=16, lambda_=r, learning_rate=0.1, batch_size=16)
        stats = regression.train_cross_validation(10)
        means_of_each.append([
            stats[:, -1, 1].mean(), stats[:, -1, 1].std(),
            stats[:, -1, 2].mean(), stats[:, -1, 2].std(),
            stats[:, -1, 4].mean(), stats[:, -1, 4].std(),
            stats[:, -1, 5].mean(), stats[:, -1, 5].std(),
            r])
    arr = np.array(means_of_each)
    # Sort by regularization so the line is ordered
    arr = arr[arr[:, 8].argsort()]
    r = arr[:, 8]

    plots.plot_train_val_vs_param(
        params=r,
        train_mean=arr[:, 0],
        val_mean=arr[:, 2],
        title="Validation CE vs lambda",
        xlabel="lambda",
        ylabel="CE loss",
    )

    plots.plot_train_val_vs_param(
        params=r,
        train_mean=arr[:, 4],
        val_mean=arr[:, 6],
        title="Validation Accuracy vs lambda",
        xlabel="lambda",
        ylabel="CE loss",
    )

def task4(d):
    # X, y as numpy arrays; y should be 1D (n,)
    y = d.trainY.ravel()
    X = d.trainX
    Cs = np.logspace(-4, 4, 30)
    coefs = []
    CV_acc = []


    for C in tqdm(Cs):
        clf = SkLogisticRegression(
            l1_ratio=1.0,
            solver="saga",
            C=C,
            max_iter=5000,
            tol=1e-4,
            warm_start=True
        )
        clf.fit(X, y)
        coefs.append(clf.coef_.ravel())

    coefs = np.array(coefs)  # shape (len(Cs), n_features)
    non_zero = np.count_nonzero(coefs, axis=1)

    k = 10

    scores = np.mean(np.abs(coefs), axis=0)
    topk_idx = np.argsort(scores)[-k:][::-1]

    coefs_topk = np.abs(coefs[:, topk_idx])
    feature_names_topk = d.X.columns[topk_idx]

    plot_sparsity(Cs, non_zero)
    plot_reg_path(Cs, coefs_topk, feature_names_topk.to_numpy())

    clf = SkLogisticRegressionCV(
        Cs=Cs,
        l1_ratios=[1.0],
        solver="saga",
        max_iter=5000,
        tol=1e-4,
        cv=6,
        scoring="accuracy",
    )
    clf.fit(X, y)
    mean_cv_acc = clf.scores_[1].mean(axis=0)
    plots.plot_cv_acc_vs_c(Cs, mean_cv_acc)




if __name__ == '__main__':
    d = Dataset()
    arr = np.load("stats_task2.npy")
    # task1(d)
    # to retrain uncomment this line - values are saved in stats_task2.npy
    # task2(d)
    # task2_best_model(d)
    # print_task2(arr)
    # task3(d)
    task4(d)



