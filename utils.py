import numpy as np
import pandas as pd
from typing import Tuple, Any
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


def cross_entropy(targets: np.ndarray, activations: np.ndarray) -> float:
    tot = np.multiply(targets, np.log(activations))
    return -tot.sum()


def sparse_cross_entropy(targets: np.ndarray, activations: np.ndarray) -> float:
    '''Assumes targets are one dimensional arrays'''
    tot = np.log(activations[np.arange(targets.shape[0]), targets])
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
    '''Computes basic Mean Squared Error'''
    sse = np.sum(np.power(targets - activations, 2))
    return sse / len(targets)


def convolution(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    '''
    Performs convolution with `image` and `kernel` where `image is a 3D tensor and `kernel` is a 4D tensor, r
    esulting in a 3D tensor. Each 2D slice of the output will be one feature map of `image`.
    
    :param image:
    :param kernel:
    :param stride:
    :param padding:
    :return:
    '''
    # Compute dimensions for the output
    output_dim1 = ((image.shape[1] + 2 * padding - kernel.shape[0]) // stride) + 1
    output_dim2 = ((image.shape[2] + 2 * padding - kernel.shape[1]) // stride) + 1
    output_dim3 = kernel.shape[3]
    output = np.zeros(shape=(output_dim1, output_dim2, output_dim3))
    if padding == 0:
        # No padding, takes advantage of numpy slicing for efficiency
        for c2 in range(output_dim3):
            for i in range(output_dim1):
                for j in range(output_dim2):
                    im_slice = image[(stride * i): (stride * i) + kernel.shape[1],
                                     (stride * j): (stride * j) + kernel.shape[2], :].flatten()
                    ker_slice = kernel[:, :, :, c2].flatten()
                    output[i, j, c2] = im_slice.dot(ker_slice)
        return output
    else:
        pad_rows = pad_cols = padding
        for c2 in range(output_dim3):
            for i in range(output_dim1):
                for j in range(output_dim2):
                    # Get kernel indices for slicing
                    start_row_k = max(pad_rows - (stride * i), 0)
                    stop_row_k = min(image.shape[0] - ((stride * i) - pad_rows), kernel.shape[0])
                    start_col_k = max(pad_cols - (stride * j), 0)
                    stop_col_k = min(image.shape[1] - ((stride * j) - pad_cols), kernel.shape[1])
                    # Get image indices for slicing
                    start_row_im = max((stride * i) - pad_rows, 0)
                    stop_row_im = min((stride * i) - pad_rows + kernel.shape[0], image.shape[0])
                    start_col_im = max((stride * j) - pad_cols, 0)
                    stop_col_im = min((stride * j) - pad_cols + kernel.shape[1], image.shape[1])
                    # Slice kernel and image to take advantage of dot product
                    im_slice = image[start_row_im: stop_row_im, start_col_im: stop_row_im, :].flatten()
                    ker_slice = kernel[start_row_k: stop_row_k, start_col_k: stop_col_k, :, c2].flatten()
                    output[i, j, c2] = im_slice.dot(ker_slice)
        return output


def backward_convolution_kernel(images: np.ndarray, delta: np.ndarray, out_shape: Tuple[int, ...], stride: int = 1, padding: int = 0) -> np.ndarray:
    '''
    Perform the necessary convolution during backpropagation. Returns the gradient with respect to the kernel.

    :param images: The batch of `images` needed to find the gradient of the loss with respect to the kernel.
    :param delta: The gradient received from the next layer.
    :param stride: Controls how many steps we shift the kernel across the image.
    :param padding: How many zeros are padded around the edges of the image. At this time padding is applied evenly
                    along all spatial dimensions of the images.
    :return: An np.ndarray that is the gradient of the with respect to the kernel
    '''
    output = np.zeros(shape=out_shape)
    N = images.shape[0]
    if not padding:
        # Perform valid convolution of some kind
        if stride == 1:
            for n in range(N):
                for c2 in range(output.shape[3]):
                    for c1 in range(output.shape[2]):
                        for i in range(output.shape[0]):
                            for j in range(output.shape[1]):
                                im_slice = images[n, i: i + delta.shape[1], j: j + delta.shape[2], c1].flatten()
                                del_slice = delta[n, :, :, c2].flatten()
                                output[i, j, c1, c2] += im_slice.dot(del_slice)
            return output
        else:
            # valid convolution with stride, rarely employed but available
            for n in range(N):
                for c2 in range(output.shape[3]):
                    for c1 in range(output.shape[2]):
                        for i in range(output.shape[0]):
                            for j in range(output.shape[1]):
                                for ii in range(delta.shape[1]):
                                    for jj in range(delta.shape[2]):
                                        output[i, j, c1, c2] += images[n, (stride * ii) + i, (stride * jj) + j, c1] * \
                                                                delta[n, ii, jj, c2]
            return output
    else:
        # We are performing convolution with padding, either full or same
        if stride == 1:
            for n in range(N):
                for c2 in range(output.shape[3]):
                    for c1 in range(output.shape[2]):
                        for i in range(output.shape[0]):
                            for j in range(output.shape[1]):
                                # Get indices for delta
                                start_row_d = max(padding - i, 0)
                                stop_row_d = min(images.shape[1] - i + padding, delta.shape[1])
                                start_col_d = max(padding - j, 0)
                                stop_col_d = min(images.shape[2] - j + padding, delta.shape[2])
                                # Get indices for image slice
                                start_row_im = max(i - padding, 0)
                                stop_row_im = min(i - padding + delta.shape[1], images.shape[1])
                                start_col_im = max(j - padding, 0)
                                stop_col_im = min(j - padding + delta.shape[2], images.shape[2])
                                # Get slices to take advantage of dot product
                                im_slice = images[n, start_row_im: stop_row_im, start_col_im: stop_col_im, c1].flatten()
                                del_slice = delta[n, start_row_d: stop_row_d, start_col_d: stop_col_d, c2].flatten()
                                output[i, j, c1, c2] += im_slice.dot(del_slice)
            return output
        else:
            for n in range(N):
                for c2 in range(delta.shape[3]):
                    for c1 in range(images.shape[3]):
                        for i in range(output.shape[0]):
                            for j in range(output.shape[1]):
                                del_sr = max(int(np.ceil((padding - i) / stride)), 0)
                                del_sc = max(int(np.ceil((padding - j) / stride)), 0)
                                row_buf = max(i - padding, 0)
                                col_buf = max(j - padding, 0)
                                im_sr = del_sr + row_buf
                                im_sc = del_sc + col_buf
                                for ii in range(delta.shape[1] - del_sr):
                                    for jj in range(delta.shape[2] - del_sc):
                                        output[i, j, c1, c2] += delta[n, del_sr + ii, del_sc + jj, c2] * \
                                                                images[n, im_sr + (stride * ii),
                                                                       im_sc + (stride * jj), c1]
            return output


def backward_convolution_inputs(kernel: np.ndarray,
                                delta: np.ndarray,
                                out_shape: Tuple[int, ...],
                                stride: int = 1,
                                padding: int = 0) -> np.ndarray:
    """
    Perform convolution during backpropagation update the delta term from
    the gradient with respect to inputs at current layer.
    :param kernel: The kernel of the current layer
    :param delta: The gradient received from the next layer
    :param out_shape: The desired shape of our output
    :param stride: Controls how many units we shift the kernel over during forward propagation,
                   will be the same value passed in as a parameter.
    :param padding: Controls the padding around the spatial dimensions of the image,
                    is applied evenly to all spatial dimensions of the image.
    :return: The gradient with respect to the inputs for further backpropagation
    """
    output = np.zeros(shape=out_shape)
    N = output.shape[0]
    if stride == 1:
        for n in range(N):
            for c1 in range(kernel.shape[2]):
                for c2 in range(kernel.shape[3]):
                    for i in range(output.shape[1]):
                        for j in range(output.shape[2]):
                            # Get indices for kernel
                            start_row_k = max(padding - i, 0)
                            stop_row_k = min(delta.shape[1] - i + padding, kernel.shape[0])
                            start_col_k = max(padding - j, 0)
                            stop_col_k = min(delta.shape[2] - j + padding, kernel.shape[1])
                            # Get indices for image slice
                            start_row_im = max(i - padding, 0)
                            stop_row_im = min(i - padding + kernel.shape[0], delta.shape[1])
                            start_col_im = max(j - padding, 0)
                            stop_col_im = min(j - padding + kernel.shape[1], delta.shape[2])
                            # Get slices to take advantage of dot product
                            delta_slice = delta[n, start_row_im: stop_row_im, start_col_im: stop_col_im, c2].flatten()
                            kernel_slice = kernel[start_row_k: stop_row_k, start_col_k: stop_col_k, c1, c2].flatten()
                            output[n, i, j, c1] += delta_slice.dot(kernel_slice)
        return output
    else:
        if padding == 0:
            for n in range(N):
                for c1 in range(kernel.shape[2]):
                    for c2 in range(kernel.shape[3]):
                        for i in range(delta.shape[1]):
                            for j in range(delta.shape[2]):
                                for ii in range(kernel.shape[0]):
                                    for jj in range(kernel.shape[1]):
                                        output[n, (stride * i) + ii, (stride * j) + jj, c1] += delta[n, i, j, c2] * \
                                                                                               kernel[ii, jj, c1, c2]
            return output
        else:
            for n in range(N):
                for c1 in range(kernel.shape[2]):
                    for c2 in range(kernel.shape[3]):
                        for i in range(output.shape[1]):
                            for j in range(output.shape[2]):
                                delta_start_row = int(np.floor((i + padding - kernel.shape[0]) / stride) + 1)
                                delta_start_col = int(np.floor((j + padding - kernel.shape[1]) / stride) + 1)
                                delta_stop_row = min(int(np.floor(((i + padding) / stride) + 1)), delta.shape[1])
                                delta_stop_col = min(int(np.floor(((j + padding) / stride) + 1)), delta.shape[2])
                                for ii in range(delta_start_row, delta_stop_row):
                                    for jj in range(delta_start_col, delta_stop_col):
                                        ker_i = i - (stride * ii - padding)
                                        ker_j = j - (stride * jj - padding)
                                        output[n, i, j, c1] += delta[n, ii, jj, c2] * kernel[ker_i, ker_j, c1, c2]
            return output


def max_pool(images: np.ndarray, filter_size:
             Tuple[int, int] = (2, 2),
             stride: int = 2,
             return_grad=True) -> Tuple[np.ndarray, ...]:

    k1, k2 = filter_size
    output_dim0 = images.shape[0]
    output_dim1 = ((images.shape[1] - k1) // stride) + 1
    output_dim2 = ((images.shape[2] - k2) // stride) + 1
    output_dim3 = images.shape[3]
    output = np.zeros(shape=(output_dim0, output_dim1, output_dim2, output_dim3))
    grad = np.zeros(shape=images.shape)
    for n in range(output_dim0):
        for c in range(output_dim3):
            for i in range(output_dim1):
                for j in range(output_dim2):
                    image_slice = images[n, (stride * i): (stride * i) + k1, (stride * j): (stride * j) + k2, c]
                    slice_max = np.max(image_slice)
                    output[n, i, j, c] = slice_max
                    # Find argmax of value to update gradient as well
                    pos = np.argmax(image_slice.flatten())
                    grad_i = pos // k2
                    grad_j = pos % k2
                    grad[n, (stride * i) + grad_i, (stride * j) + grad_j, c] = 1

    if return_grad:
        return output, grad
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














