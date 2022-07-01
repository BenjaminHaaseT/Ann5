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


def binary_cross_entropy_with_logits(targets: np.ndarray, activations: np.ndarray) -> float:
    # Perfroms bce with logits
    pass


def mse(targets: np.ndarray, activations: np.ndarray) -> float:
    '''Computes basice Mean Squared Error'''
    sse = np.sum(np.power(targets - activations, 2))
    return sse / len(targets)


def convolution(image: np.ndarray, kernel: np.ndarray, mode: str = "valid", stride: int = 1) -> np.ndarray:
    """
    Performs convolution on a 3-d image and 4-d tensor `kernel`, to produce a 3-d output
    i.e. a set of 2-d feature maps.

    As it stands image dimension and kernel dimension must be appropriate
    in order to output meaningful results using stride.
    In other words the dimensions of the image minus dimensions of kernel must be divisible by stride.

    To be used for convolutional neural networks
    :param image:
    :param kernel:
    :param mode:
    :return:
    """

    if mode == "valid":
        output_dim1 = ((image.shape[0] - kernel.shape[0]) // stride) + 1
        output_dim2 = ((image.shape[1] - kernel.shape[1]) // stride) + 1
        output_dim3 = kernel.shape[3]
        output = np.zeros(shape=(output_dim1, output_dim2, output_dim3))
        for c in range(output.shape[2]):
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    im_slice = image[(stride * i): (stride * i) + kernel.shape[0],
                               (stride * j): (stride * j) + kernel.shape[1], :].flatten()
                    kernel_slice = kernel[:, :, :, c].flatten()
                    output[i, j, c] = im_slice.dot(kernel_slice)

        return output

    elif mode == "same":
        pad_rows = (kernel.shape[0] - 1) // 2
        pad_cols = (kernel.shape[1] - 1) // 2
        output_dim1 = ((image.shape[0] - kernel.shape[0] + 2 * pad_rows) // stride) + 1
        output_dim2 = ((image.shape[1] - kernel.shape[1] + 2 * pad_cols) // stride) + 1
        output_dim3 = kernel.shape[3]
        output = np.zeros((output_dim1, output_dim2, output_dim3))
        # If stride is one
        if stride == 1:
            for c in range(output.shape[2]):
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        start_row_k = max(pad_rows - i, 0)
                        stop_row_k = min(output.shape[0] + pad_rows - i, kernel.shape[0])
                        start_col_k = max(pad_cols - j, 0)
                        stop_col_k = min(output.shape[1] + pad_cols - j, kernel.shape[1])
                        # Get indices for slicing image
                        start_row_im = max(i - pad_rows, 0)
                        stop_row_im = min(kernel.shape[0] - pad_rows + i, output.shape[0])
                        start_col_im = max(j - pad_cols, 0)
                        stop_col_im = min(kernel.shape[1] - pad_cols + j, output.shape[1])
                        im_slice = image[start_row_im: stop_row_im, start_col_im: stop_col_im, :].flatten()
                        ker_slice = kernel[start_row_k: stop_row_k, start_col_k: stop_col_k, :, c].flatten()
                        output[i, j, c] = im_slice.dot(ker_slice)

            return output

        elif stride >= 2:
            for c in range(output.shape[2]):
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        # Get indices for slicing kernel
                        start_row_k = max(pad_rows - (stride * i), 0)
                        stop_row_k = min(image.shape[0] - ((stride * i) - pad_rows), kernel.shape[0])
                        start_col_k = max(pad_cols - (stride * j), 0)
                        stop_col_k = min(image.shape[1] + ((stride * j) - pad_cols), kernel.shape[1])
                        # Get indices for slicing image
                        start_row_im = max((stride * i) - pad_rows, 0)
                        stop_row_im = min(((stride * i) - pad_rows) + kernel.shape[0], image.shape[0])
                        start_col_im = max((stride * j) - pad_cols, 0)
                        stop_col_im = min((stride * j) - pad_cols + kernel.shape[1], image.shape[1])
                        # Slice image and kernel, then flatten to take advantage of dot product
                        im_slice = image[start_row_im: stop_row_im, start_col_im: stop_col_im, :].flatten()
                        ker_slice = kernel[start_row_k: stop_row_k, start_col_k: stop_col_k, :, c].flatten()
                        output[i, j, c] = im_slice.dot(ker_slice)

            return output


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














