from common.common import cross_entropy_loss, constraint_penalty


class SimpleDNN:
    def __init__(self, layers, regularization=None):
        self.layers = layers
        self.regularization = regularization

    def feedforward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
        return x

    def backpropagate(self, x, y, learning_rate):
        output_error = self.feedforward(x) - y
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagate(X, y, learning_rate)
            #if epoch % 10 == 0:
            print(epoch)
            y_pred = self.feedforward(X)
            penalty = constraint_penalty(X, y_pred)
            loss = cross_entropy_loss(y, y_pred)
            if loss <= 0.07:
                return
            total_loss = loss + penalty
            print(f"Epoch {epoch} - Loss: {loss}")
