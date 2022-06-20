import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-a))


def sigmoid_grad(a: np.ndarray) -> np.ndarray:
    return np.multiply(a, np.ones_like(a) - a)


def tanh_grad(a: np.ndarray) -> np.ndarray:
    return np.ones_like(a) - np.power(a, 2)


def relu(a: np.ndarray) -> np.ndarray:
    return np.multiply(a, a > 0)


def relu_grad(a: np.ndarray) -> np.ndarray:
    return a > 0


def leaky_relu(a: np.ndarray, scalar: float) -> np.ndarray:
    neo = np.where(a > 0, 1, scalar)
    return np.multiply(a, neo)


def leaky_relu_grad(a: np.ndarray, scalar: float) -> np.ndarray:
    return np.where(a > 0, 1, scalar)


def softmax(a: np.ndarray) -> np.ndarray:
    expA = np.exp(a)
    return expA / np.sum(expA, axis=1, keepdims=True)


def classification_rate(targets: np.ndarray, predictions: np.ndarray) -> float:
    return np.sum(targets == predictions) / len(targets)


def error_rate(targets: np.ndarray, predictions: np.ndarray) -> float:
    return 1 - classification_rate(targets, predictions)


def cross_entropy(targets: np.ndarray, p_y: np.ndarray) -> float:
    tot = np.multiply(targets, np.log(p_y))
    return -tot.sum()


def sparse_cross_entropy(targets: np.ndarray, p_y: np.ndarray) -> float:
    '''Assumes targets are one dimensional arrays'''
    tot = np.log(p_y[np.arange(targets.shape[0]), targets])
    return -np.mean(tot)


def binary_cross_entropy(targets: np.ndarray, activations: np.ndarray) -> float:
    '''Performs binary cross entropy'''
    activations = np.ravel(activations)
    tot = np.multiply(targets, np.log(activations)) + \
          np.multiply(np.ones_like(targets) - targets, np.log(np.ones_like(activations) - activations))
    return -np.mean(tot)


def mse(targets: np.ndarray, y_hat: np.ndarray) -> float:
    '''Computes basice Mean Squared Error'''
    sse = np.sum(np.power(targets - y_hat, 2))
    return sse / len(targets)


def get_spirals():
    '''Get spirals for testing classification'''
    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])
    # Inputs
    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()
    # Add noise
    X += np.random.randn(600, 2) * 0.5
    Y = np.array([0] * 100 + [1] * 100 + [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100)
    return X, Y


def get_spiral(show_fig=False):
    '''Generate Spiral data for classification'''
    radi_1 = np.linspace(0.15, 7, 15000)
    radi_2 = radi_1
    thetas_1 = np.linspace(0, 4 * np.pi, 15000)
    thetas_2 = np.linspace(np.pi, 5 * np.pi, 15000)
    noise_1, noise_2, noise_3, noise_4 = np.random.randn(15000), np.random.randn(15000), \
                                         np.random.randn(15000), np.random.randn(15000)
    x_1, x_2 = radi_1 * np.cos(thetas_1), radi_1 * np.sin(thetas_1)
    x_3, x_4 = radi_2 * np.cos(thetas_2), radi_2 * np.sin(thetas_2)
    x_1 = x_1 + (noise_1 * 0.1)
    x_2 = x_2 + (noise_2 * 0.1)
    x_3 = x_3 + (noise_3 * 0.1)
    x_4 = x_4 + (noise_4 * 0.1)
    class_1 = np.concatenate((x_1.reshape(len(x_1), 1), x_2.reshape(len(x_2), 1)), axis=1)
    class_2 = np.concatenate((x_3.reshape(len(x_3), 1), x_4.reshape(len(x_4), 1)), axis=1)
    data = np.vstack((class_1, class_2))
    labels = np.array([0] * len(class_1) + [1] * len(class_2))
    if show_fig:
        plt.scatter(x_1, x_2, alpha=0.1, color='purple')
        plt.scatter(x_3, x_4, alpha=0.1, color='orange')
        plt.show()
    return data, labels


def get_mnist(normalize=True):
    print("Reading in and transforming data...")

    df = pd.read_csv('/Users/benjaminhaase/development/Personal/Python/ModernDeepLearning/mnist_train.csv')
    data = df.to_numpy().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]
    X_train, X_validate = X[:-1000, :], X[-1000:, :]
    Y_train, Y_validate = Y[:-1000], Y[-1000:]

    # Normalize data
    if normalize:
        mu = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        # Where are the all zero
        idx = np.where(std == 0)[0]
        assert(np.all(std[idx] == 0))
        np.place(std, std == 0, 1)

        X_train = (X_train - mu) / std
        X_validate = (X_validate - mu) / std

    print("Successfully read in data")
    return X_train, X_validate, Y_train.astype(np.int32), Y_validate.astype(np.int32)














