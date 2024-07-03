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

        # Apply symbolic constraint to identify potential attack patterns
        penalty_scores = self.symbolic_constraint(input_data)

        adjusted_output = self.output * (1 - penalty_scores[:, None])  # Assuming penalty_scores shape is (n_samples,)

        return adjusted_output

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


    def symbolic_constraint(self, input_data):
        """
        Identifies patterns indicative of attack traffic based on input features.

        Parameters:
        - input_data: np.array, shape (n_samples, n_features)

        Returns:
        - penalty_scores: np.array, shape (n_samples,), penalty scores for attack patterns
        """
        # Define thresholds based on domain knowledge
        flow_duration_threshold = 100000
        tot_fwd_pkts_threshold = 50
        totlen_fwd_pkts_low_threshold = 500

        # Extract relevant features from input_data
        flow_duration = input_data[:, 0]  # Assuming 'Flow Duration' is the first feature
        tot_fwd_pkts = input_data[:, 1]  # Assuming 'Tot Fwd Pkts' is the second feature
        totlen_fwd_pkts = input_data[:, 3]  # Assuming 'TotLen Fwd Pkts' is the fourth feature

        #print(flow_duration)

        # Apply thresholds to identify potential attack patterns
        is_attack = np.logical_and.reduce([
            flow_duration > flow_duration_threshold,
            tot_fwd_pkts > tot_fwd_pkts_threshold,
            totlen_fwd_pkts < totlen_fwd_pkts_low_threshold
        ])

        # Convert boolean array to float (or int) to represent penalty scores
        penalty_scores = is_attack.astype(float)

        return penalty_scores