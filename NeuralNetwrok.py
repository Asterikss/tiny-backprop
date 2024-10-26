import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def train(self, x, y, epochs, learning_rate, batch_size):
        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            total_loss = 0
            correct_preds = 0
            total_preds = 0

            batches = self.create_batches(x, y, batch_size)

            for batch_x, batch_y in batches:
                logits = self.forward(batch_x)

                loss = self.compute_loss(logits, batch_y)

                total_loss += loss
                predictions = self.predict(batch_x)
                correct_preds += np.sum(predictions == np.argmax(batch_y, axis=1))
                total_preds += len(batch_x)

                self.backward(self.loss_derivative(logits, batch_y), learning_rate)

            epoch_loss = total_loss / len(batches)
            epoch_accuracy = correct_preds / total_preds
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        return history

    def compute_loss(self, logits, y_true):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        m = y_true.shape[0]
        epsilon = 1e-8
        return -np.sum(y_true * np.log(probs + epsilon)) / m

    def loss_derivative(self, logits, y_true):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        return probs - y_true # Softmax and cross-entropy gradient

    def create_batches(self, X, y, batch_size):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        n_batches = len(X) // batch_size
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batches.append((X[start:end], y[start:end]))

        # Handle the case where the last batch is smaller than batch_size
        if len(X) % batch_size != 0:
            start = n_batches * batch_size
            batches.append((X[start:], y[start:]))

        return batches
