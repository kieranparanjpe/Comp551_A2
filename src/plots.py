import matplotlib.pyplot as plt

def display_epoch_train_test(data, title, validation=False):
    x, y1, y2, y3 = zip(*data)

    plt.plot(x, y1, label="train")
    if validation:
        plt.plot(x, y2, label="validation")
    plt.plot(x, y3, label="test")


    plt.xlabel("epoch")
    plt.ylabel("ce loss")
    plt.title(title)
    plt.legend()
    plt.show()
