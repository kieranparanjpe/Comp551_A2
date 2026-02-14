import matplotlib.pyplot as plt

def display_epoch_train_test(x, ys, title, validation=False, yAxis="ce loss"):
    y1, y2, y3 = zip(*ys)

    plt.plot(x, y1, label="train")
    if validation:
        plt.plot(x, y2, label="validation")
    plt.plot(x, y3, label="test")


    plt.xlabel("epoch")
    plt.ylabel(yAxis)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_train_val_vs_param(params, train_mean, val_mean,
                            title=None, xlabel="lambda", ylabel="ce loss"):
    plt.xscale("log")

    plt.plot(params, train_mean, label="train")

    plt.plot(params, val_mean, label="validation")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_sparsity(Cs, number_non_zero):
    plt.xscale("log")

    plt.plot(Cs, number_non_zero)

    plt.xlabel("C")
    plt.ylabel("Number of non zero params")
    plt.title("Task 4 sparsity plot")
    plt.show()

def plot_reg_path(Cs, coef_values, col_names):
    plt.xscale("log")

    for c in range(coef_values.shape[1]):
        plt.plot(Cs, coef_values[:, c], label=col_names[c])

    plt.xlabel("C")
    plt.ylabel("abs coefficient value")
    plt.title("Task 4 reg path")
    plt.legend()
    plt.show()

def plot_cv_acc_vs_c(Cs, cv_acc_mean):
    plt.xscale("log")

    plt.plot(Cs, cv_acc_mean)

    plt.xlabel("C")
    plt.ylabel("cv accuracy")
    plt.title("Task 4 cv performance vs c")
    plt.show()
