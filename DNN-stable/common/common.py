import numpy as np


# Activation functions and their derivatives
def sigmoid(x):
    epsilon = 1e-7  # Small constant for numerical stability
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Mean Squared Error Loss Function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def cross_entropy_loss(y_true, y_pred):
    # Small epsilon value to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def l1_regularization(weights, l1_lambda=0.01):
    return l1_lambda * np.sign(weights)


def l2_regularization(weights, l2_lambda=0.01):
    return l2_lambda * weights


def constraint_penalty(X, predictions):
    # Assuming 'Flow Duration' is at index 0, 'Tot Fwd Pkts' at index 1, and 'Tot Bwd Pkts' at index 2 in X
    flow_duration = X[:, 0]
    tot_fwd_pkts = X[:, 1]
    tot_bwd_pkts = X[:, 2]

    epsilon = 1e-5
    lambda_penalty = 0.01  # Example scaling factor; adjust based on your needs

    penalty_per_sample = lambda_penalty * (flow_duration / (tot_fwd_pkts + tot_bwd_pkts + epsilon))
    penalty = np.mean(penalty_per_sample)  # Calculate mean penalty across all samples

    return penalty
