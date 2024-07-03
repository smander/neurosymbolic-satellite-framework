import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation_function, activation_derivative,
                 regularization_function=None):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.regularization_function = regularization_function

    def forward(self, input_data):
        self.input_data = input_data
        self.output = self.activation_function(np.dot(input_data, self.weights) + self.biases)
        return self.output

    def backward(self, output_error, learning_rate):
        # Calculate local gradient
        local_gradient = self.activation_derivative(self.output)

        # Update output error with local gradient for weight update
        output_error *= local_gradient

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input_data.T, output_error)

        # Apply regularization if a regularization function is provided
        if self.regularization_function is not None:
            regularization_term = self.regularization_function(self.weights)
            weights_error += regularization_term

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error
