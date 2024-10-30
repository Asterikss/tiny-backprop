import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt


def load_dataset(normalize=True, verbose=False):
    iris = datasets.load_iris()
    X = iris.data # type: ignore[reportAttributeAccessIssue]
    y = iris.target # type: ignore[reportAttributeAccessIssue]

    if verbose:
        print("Original dataset shape:")
        print("X: ", np.shape(X))
        print("y: ", np.shape(y))
        print("X: ", X[:3])
        print("y: ", y[:3])

    num_classes = len(set(y))

    if verbose:
        print("Number of classes: ", num_classes)

    one_hot = np.zeros((len(y), num_classes))

    for i in range(len(one_hot)):
        one_hot[i][y[i]] = 1

    if normalize:
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        X = (X - mean) / std

    if verbose:
        print("Dataset shape after transformation:")
        print("X: ", np.shape(X))
        print("y: ", np.shape(one_hot))
        print("X: ", X[:3])
        print("y: ", one_hot[:3])
        if normalize:
            print("Dataset has been normalized by columns")

    return train_test_split(X, one_hot, test_size=0.33)


def test(model, X_test, y_test):
    # expects y_test to be a onehot
    assert(len(X_test) == len(y_test))
    print("------------------------------")
    predictions = model.predict(X_test)
    result = np.sum(predictions == y_test.argmax(axis=1))
    print(f"Result: {result}/{len(y_test)}. Accuracy: {result/len(y_test)}")
    print("------------------------------")


def plot_metrics(history):
    epochs = len(history['loss'])
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['loss'], label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['accuracy'], label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
