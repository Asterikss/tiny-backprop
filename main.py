from NeuralNetwrok import NeuralNetwork
from Layer import Layer
from utils import load_dataset, test, plot_metrics


def main():
    model = NeuralNetwork()
    model.add(Layer(4, 10))
    model.add(Layer(10, 10))
    model.add(Layer(10, 3))

    X_train, X_test, y_train, y_test = load_dataset(normalize=True, verbose=False)

    print("train set test before training")
    test(model, X_train, y_train)
    print("test set test before training")
    test(model, X_test, y_test)

    # history = model.train(X_train, y_train, epochs=300, learning_rate=0.0001, batch_size=len(X_train))
    # history = model.train(X_train, y_train, epochs=1000, learning_rate=0.001, batch_size=20)
    history = model.train(X_train, y_train, epochs=500, learning_rate=0.001, batch_size=20)


    print("train set statistics")
    test(model, X_train, y_train)
    print("test set statistics:")
    test(model, X_test, y_test)

    plot_metrics(history)


if __name__ == "__main__":
    main()
