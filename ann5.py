import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import math as m
from sklearn.utils import shuffle
from typing import Callable, List, Tuple


ACTIVATION_FUNCTIONS = {
    'sigmoid': ut.sigmoid,
    'tanh': np.tanh,
    'relu': ut.relu,
    'leakyrelu': ut.leaky_relu,
    'softmax': ut.softmax,
}

GRADIENT_FUNCTIONS = {
    'sigmoid': ut.sigmoid_grad,
    'tanh': ut.tanh_grad,
    'relu': ut.relu_grad,
    'leakyrelu': ut.leaky_relu_grad,
}

MOMENTUM_UPDATERS = {
    'standard': (lambda velocity, differential, mu: velocity),
    'nesterov': (lambda velocity, differential, mu: (mu * velocity) - differential)
}

WEIGHT_INITIALIZERS = {
    ''
}


class BaseModule(object):
    '''Base class for all Layer/Module objects'''
    def __init__(self):
        pass

    def forward(self, forward_input: np.ndarray, is_training=True):
        pass


class DifferentiableModule(BaseModule):
    '''Base layer for any class that will contribute to back propagation'''
    def __init__(self):
        super().__init__()

    def update_delta(self, delta: np.ndarray) -> np.ndarray:
        pass


class ParameterModule(DifferentiableModule):
    '''Base class for all modules/layers that have parameters'''
    def __init__(self):
        super().__init__()

    def get_gradients(self, delta: np.ndarray, reg: float) -> List[np.ndarray]:
        pass

    def params(self) -> List[np.ndarray]:
        pass


class InputLayer(ParameterModule):
    '''Base class for all input layers'''
    def __init__(self):
        super().__init__()


class InputLinearLayer(InputLayer):
    '''Input layer that performs a linear transformation'''
    def __init__(self, n_in: int, n_out: int, bias=True):
        super().__init__()
        if n_out > 1:
            self.weights = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))
            self.input_ = None
        else:
            self.weights = np.random.randn(n_in) * np.sqrt(2 / (n_in + n_out))
        if bias:
            self.bias = np.zeros(n_out)
            self.parameters = [self.weights, self.bias]
        else:
            self.parameters = [self.weights]

    def forward(self, forward_input: np.ndarray, is_training=True):
        if is_training:
            self.input_ = forward_input
        if len(self.parameters) == 2:
            return forward_input.dot(self.weights) + self.bias
        else:
            return forward_input.dot(self.weights)

    def get_gradients(self, delta: np.ndarray, reg: float) -> List[np.ndarray]:
        grad_w = self.input_.T.dot(delta) + reg * self.weights
        if len(self.parameters) == 2:
            grad_b = delta.sum(axis=0) + reg * self.bias
            return [grad_w, grad_b]
        return [grad_w]

    def params(self) -> List[np.ndarray]:
        return self.parameters


class LinearLayer(InputLinearLayer):
    '''For a hidden linear layer'''
    def __init__(self, n_in: int, n_out: int, bias=True):
        super().__init__(n_in, n_out, bias)

    def update_delta(self, delta: np.ndarray) -> np.ndarray:
        return delta.dot(self.weights.T)


class Dropout(BaseModule):
    '''Class that provides dropout regularization'''
    def __init__(self, p_keep: float):
        super().__init__()
        self.p_keep = p_keep

    def forward(self, forward_input: np.ndarray, is_training=True):
        if is_training:
            # Create mask and pass forward
            mask = np.random.binomial(n=1, p=self.p_keep, size=forward_input.shape).reshape(forward_input.shape)
            return np.multiply(forward_input, mask)
        else:
            return self.p_keep * forward_input


class BatchNormalization(ParameterModule):
    '''Provides batch-normalization'''
    def __init__(self, size: int, decay=0.99, epsilon=10e-8):
        super().__init__()
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)
        self.parameters = [self.gamma, self.beta]
        self.decay = decay
        self.epsilon = epsilon
        self._input = None
        self.first = True
        self.running_mean = 0
        self.running_var = 0
        self.batch_mean = 0
        self.batch_var = 0

    def forward(self, forward_input: np.ndarray, is_training=True):
        if is_training:
            # Perform batch normalization during training
            self.batch_mean = forward_input.mean(axis=0)
            self.batch_var = forward_input.var(axis=0)
            # Update running mean/var
            if self.first:
                self.running_mean = self.batch_mean
                self.running_var = self.batch_var
                self.first = False
            else:
                self.running_mean = (self.decay * self.running_mean) + ((1 - self.decay) * self.batch_mean)
                self.running_var = (self.decay * self.running_var) + ((1 - self.decay) * self.batch_var)
            # Standardize input, save for backpropagation
            self._input = (forward_input - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            return (self.gamma * self._input) + self.beta
        else:
            z_hat_test = (forward_input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return (self.gamma * z_hat_test) + self.beta

    def get_gradients(self, delta: np.ndarray, reg: float) -> List[np.ndarray]:
        assert(delta.shape == self._input.shape)
        grad_gamma = np.multiply(delta, self._input).sum(axis=0) + reg * self.gamma
        grad_beta = delta.sum(axis=0) + reg * self.beta
        return [grad_gamma, grad_beta]

    def update_delta(self, delta: np.ndarray) -> np.ndarray:
        assert(delta.shape[1] == self.gamma.shape[0])
        return (delta * self.gamma) / np.sqrt(self.batch_var + self.epsilon)

    def params(self) -> List[np.ndarray]:
        return self.parameters


class ActivationFunction(DifferentiableModule):
    '''Wrapper class for an activation function'''
    def __init__(self, activation_function: str):
        super().__init__()
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        self.gradient_function = GRADIENT_FUNCTIONS[activation_function]
        self.activations = None

    def forward(self, forward_input: np.ndarray, is_training=True) -> np.ndarray:
        if is_training:
            self.activations = self.activation_function(forward_input)
            return self.activations
        return self.activation_function(forward_input)

    def update_delta(self, delta: np.ndarray) -> np.ndarray:
        return np.multiply(delta, self.gradient_function(self.activations))


class Sigmoid(ActivationFunction):
    '''Wrapper class for sigmoid activation function'''
    def __init__(self, is_final=False):
        super().__init__('sigmoid')
    #     self.is_final = is_final
    #
    # def update_delta(self, delta: np.ndarray) -> np.ndarray:
    #     if self.is_final:
    #         return np.multiply(delta, self.gradient_function(np.ravel(self.activations)))
    #     return np.multiply(delta, self.gradient_function(self.activations))



class Tanh(ActivationFunction):
    '''Wrapper class for tanh activation function'''
    def __init__(self):
        super().__init__('tanh')


class ReLU(ActivationFunction):
    '''Wrapper class for tanh activation function'''
    def __init__(self):
        super().__init__('relu')


class LeakyReLU(ActivationFunction):
    '''Wrapper class for tanh activation function'''
    def __init__(self, scalar=0.05):
        super().__init__('leakyrelu')
        self.scalar = scalar

    def forward(self, forward_input: np.ndarray, is_training=True) -> np.ndarray:
        if is_training:
            self.activations = self.activation_function(forward_input, self.scalar)
            return self.activations
        return self.activation_function(forward_input, self.scalar)

    def update_delta(self, delta: np.ndarray) -> np.ndarray:
        return np.multiply(delta, self.gradient_function(self.activations, self.scalar))


class FinalActivationFunction(BaseModule):
    '''Base class for any final activation function, not differentiable
    because gradient w.r.t final activation function will be computed in objective function'''
    def __init__(self):
        super().__init__()


class Softmax(FinalActivationFunction):
    '''Wrapper class for softmax function. To be used as a final activation function.'''
    def __init__(self):
        super().__init__()

    def forward(self, forward_input: np.ndarray, is_training=True) -> np.ndarray:
        return ut.softmax(forward_input)


class ObjectiveFunction(object):
    '''Base class for an objective function'''
    def __init__(self):
        pass

    def loss(self, activations: np.ndarray, targets: np.ndarray) -> float:
        pass

    def predict(self, activations: np.ndarray) -> np.ndarray:
        pass

    def get_delta(self, activations: np.ndarray, targets: np.ndarray) -> np.ndarray:
        pass


class SparseCategoricalCrossEntropy(ObjectiveFunction):
    '''Performs K class classification with categorical cross entropy,
    assumes all activations are from the softmax activation function'''
    def __init__(self):
        super().__init__()

    def loss(self, activations: np.ndarray, targets: np.ndarray) -> float:
        return ut.sparse_cross_entropy(targets, activations)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return np.ravel(np.argmax(activations, axis=1))

    def get_delta(self, activations: np.ndarray, targets: np.ndarray) -> np.ndarray:
        activations[np.arange(targets.shape[0]), targets] -= np.ones_like(targets)
        return activations


class BinaryCrossEntropy(ObjectiveFunction):
    '''For binary classification, assumes inputs are from the Sigmoid'''
    def __init__(self):
        super().__init__()

    def loss(self, activations: np.ndarray, targets: np.ndarray) -> float:
        return ut.binary_cross_entropy(targets, activations)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return np.round(activations)

    def get_delta(self, activations: np.ndarray, targets: np.ndarray) -> np.ndarray:
        activations = np.ravel(activations)
        return ((np.ones_like(targets) - targets) / (np.ones_like(activations) - activations)) - (targets / activations)


class LearningRateScheduler(object):
    '''Base class for a learning rate scheduler, to aid learning rate scheduling'''
    def __init__(self):
        pass

    def set_up(self, layers: List[BaseModule]) -> None:
        pass

    def update_cache(self, layer_index: int, gradients: List[np.ndarray]) -> None:
        pass

    def get_cache_updates(self, layer_index: int, gradients: List[np.ndarray]) -> List[np.ndarray]:
        pass


# class ExponentialLearningRateScheduler(LearningRateScheduler):


class AdaGrad(LearningRateScheduler):
    '''Provides AdaGrad adaptive learning rate functionality'''
    def __init__(self, epsilon=10e-8):
        super().__init__()
        self.epsilon = epsilon
        self.cache = {}

    def set_up(self, layers: List[BaseModule]) -> None:
        for i in range(len(layers)):
            # If parameter model add cache
            if isinstance(layers[i], ParameterModule):
                current_cache = {}
                parameters = layers[i].params()
                for j in range(len(parameters)):
                    current_cache[j] = np.ones_like(parameters[j])
                # Finally add cache to 'self.cache'
                self.cache[i] = current_cache

    def update_cache(self, layer_index: int, gradients: List[np.ndarray]) -> None:
        for j in range(len(gradients)):
            self.cache[layer_index][j] += np.power(gradients[j], 2)

    def get_cache_updates(self, layer_index: int, gradients: List[np.ndarray]) -> List[np.ndarray]:
        return [
            gradients[j] / np.sqrt(self.cache[layer_index][j] + self.epsilon) for j in range(len(gradients))
        ]


class RMSProp(LearningRateScheduler):
    '''Provides RMS Prop learning rate scheduling'''
    def __init__(self, decay=0.999, epsilon=10e-8):
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.cache = {}

    def set_up(self, layers: List[BaseModule]) -> None:
        for i in range(len(layers)):
            # If is a parameter module, then initialize lr
            if isinstance(layers[i], ParameterModule):
                initial_learning_rates = {}
                parameters = layers[i].params()
                for j in range(len(parameters)):
                    initial_learning_rates[j] = np.ones_like(parameters[j])
                # Add initial lr to 'self.learning_rates'
                self.cache[i] = initial_learning_rates

    def update_cache(self, layer_index: int, gradients: List[np.ndarray]) -> None:
        # Iterate over gradients
        for j in range(len(gradients)):
            self.cache[layer_index][j] = (self.decay * self.cache[layer_index][j]) + \
                                         ((1 - self.decay) * np.power(gradients[j], 2))

    def get_cache_updates(self, layer_index: int, gradients: np.ndarray) -> List[np.ndarray]:
        return [
            gradients[j] / np.sqrt(self.cache[layer_index][j] + self.epsilon) for j in range(len(gradients))
        ]


class Optimizer(object):
    '''Base class for all optimizer objects, provide specific optimization algorithms for backpropagation'''
    def __init__(self):
        pass

    def set_up(self, layers: List[BaseModule]) -> None:
        pass

    def optimize(self, delta: np.ndarray, layers: List[BaseModule], lr: float, reg: float):
        pass


class VanillaOptimizer(Optimizer):
    '''Provides standard "vannilla" gradient decent'''
    def __init__(self):
        super().__init__()

    def perform_parameter_updates(self, layer: ParameterModule, delta: np.ndarray, lr: float, reg: float) -> None:
        '''Helper method, to eliminate repeated code'''
        gradients = layer.get_gradients(delta, reg)
        parameters = layer.params()
        for j in range(len(parameters)):
            parameters[j] -= lr * gradients[j]

    def optimize(self, delta: np.ndarray, layers: List[BaseModule], lr: float, reg: float):
        for i in range(len(layers) - 1, 0, -1):
            # Update parameters if necessary
            if isinstance(layers[i], ParameterModule):
                self.perform_parameter_updates(layers[i], delta, lr, reg)

            # If differentiable, udpate delta term
            if isinstance(layers[i], DifferentiableModule):
                delta = layers[i].update_delta(delta)

        # Finally update input layer if necessary
        if isinstance(layers[0], ParameterModule):
            self.perform_parameter_updates(layers[0], delta, lr, reg)


class MomentumOptimizer(Optimizer):
    '''Provides gradient descent with momentum'''
    def __init__(self, momentum='standard', mu=0.9):
        super().__init__()
        self.momentum_updater = MOMENTUM_UPDATERS[momentum]
        self.mu = mu
        self.velocities = {}

    def set_up(self, layers: List[BaseModule]) -> None:
        # Set up initial velocities for each layer that has parameters i.e. 'ParameterModule' in 'layers
        for i in range(len(layers)):
            if isinstance(layers[i], ParameterModule):
                layer_velocity = {}
                parameters = layers[i].params()
                for j in range(len(parameters)):
                    # Add velocity for each parameter of the layer
                    # Initialize to 0's
                    layer_velocity[j] = np.zeros_like(parameters[j])
                # Add 'layer_velocity' to self.velocities for back propagation
                self.velocities[i] = layer_velocity

    def perform_parameter_updates(self, layer: ParameterModule, layer_index: int, delta: np.ndarray, lr: float, reg: float) -> None:
        '''Helper method to prevent duplicated code'''
        gradients = layer.get_gradients(delta, reg)
        parameters = layer.params()
        # Update velocity at current layer for each parameter 'j'
        for j in range(len(gradients)):
            self.velocities[layer_index][j] = (self.mu * self.velocities[layer_index][j]) - (lr * gradients[j])
        # Update parameters with velocity
        for j in range(len(parameters)):
            parameters[j] += self.momentum_updater(
                velocity=self.velocities[layer_index][j],
                differential=lr*gradients[j],
                mu=self.mu
            )

    def optimize(self, delta: np.ndarray, layers: List[BaseModule], lr: float, reg: float):
        for i in range(len(layers) - 1, 0, -1):
            # If parameter module, update parameters
            if isinstance(layers[i], ParameterModule):
                self.perform_parameter_updates(
                    layer=layers[i],
                    layer_index=i,
                    delta=delta,
                    lr=lr,
                    reg=reg
                )

            # if differentiable, update delta term
            if isinstance(layers[i], DifferentiableModule):
                delta = layers[i].update_delta(delta)

        # Finally update input layer if needed
        if isinstance(layers[0], ParameterModule):
            self.perform_parameter_updates(
                layer=layers[0],
                layer_index=0,
                delta=delta,
                lr=lr,
                reg=reg
            )


class StandardOptimizer(Optimizer):
    '''Optimizer that provides both momentum and adaptive learning rate capabilities'''
    def __init__(self, lr_scheduler: LearningRateScheduler, momentum="standard", mu=0.9):
        super().__init__()
        self.momentum_updater = MOMENTUM_UPDATERS[momentum]
        self.mu = mu
        self.scheduler = lr_scheduler
        self.velocities = {}

    def set_up(self, layers: List[BaseModule]) -> None:
        for i in range(len(layers)):
            if isinstance(layers[i], ParameterModule):
                layer_velocity = {}
                parameters = layers[i].params()
                for j in range(len(parameters)):
                    layer_velocity[j] = np.zeros_like(parameters[j])
                self.velocities[i] = layer_velocity
        self.scheduler.set_up(layers)

    def perform_parameter_updates(self, layer: ParameterModule, layer_index: int, delta: np.ndarray, lr: float, reg: float) -> None:
        gradients = layer.get_gradients(delta, reg)
        parameters = layer.params()
        # Update lr scheduler, then get lr updates
        self.scheduler.update_cache(
            layer_index=layer_index,
            gradients=gradients
        )

        updates = self.scheduler.get_cache_updates(
            layer_index=layer_index,
            gradients=gradients
        )

        # Update velocity next
        for j in range(len(updates)):
            self.velocities[layer_index][j] = (self.mu * self.velocities[layer_index][j]) - lr * updates[j]

        # Finally update parameters with velocity
        for j in range(len(parameters)):
            parameters[j] += self.momentum_updater(
                velocity=self.velocities[layer_index][j],
                differential=lr*updates[j],
                mu=self.mu
            )

    def optimize(self, delta: np.ndarray, layers: List[BaseModule], lr: float, reg: float):
        # Perform backpropagation
        for i in range(len(layers) - 1, 0, -1):
            # If parameter module update parameters
            if isinstance(layers[i], ParameterModule):
                self.perform_parameter_updates(
                    layer=layers[i],
                    layer_index=i,
                    delta=delta,
                    lr=lr,
                    reg=reg
                )

            # If differentiable module update delta term accordingly
            if isinstance(layers[i], DifferentiableModule):
                delta = layers[i].update_delta(delta)

        # Finally update input layer if necessary
        if isinstance(layers[0], ParameterModule):
            self.perform_parameter_updates(
                layer=layers[0],
                layer_index=0,
                delta=delta,
                lr=lr,
                reg=reg
            )


class AdamOptimizer(Optimizer):
    '''Provides ADAM optimization functionality'''
    def __init__(self, beta_1=0.99, beta_2=0.999, epsilon=10e-8):
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 1
        self.first_moments = {}
        self.second_moments = {}

    def set_up(self, layers: List[BaseModule]) -> None:
        for i in range(len(layers)):
            if isinstance(layers[i], ParameterModule):
                # Initiate first and second moments for layer
                first_moment = {}
                second_moment = {}
                parameters = layers[i].params()
                for j in range(len(parameters)):
                    first_moment[j] = np.zeros_like(parameters[j])
                    second_moment[j] = np.ones_like(parameters[j])
                self.first_moments[i] = first_moment
                self.second_moments[i] = second_moment

    def perform_parameter_updates(self, layer: ParameterModule, layer_index: int, delta: np.ndarray, lr: float, reg: float) -> None:
        gradients = layer.get_gradients(delta, reg)
        parameters = layer.params()
        # Update first moments
        for j in range(len(gradients)):
            self.first_moments[layer_index][j] = (self.beta_1 * self.first_moments[layer_index][j]) + \
                                                 ((1 - self.beta_1) * gradients[j])
        # Update second moments
        for j in range(len(gradients)):
            self.second_moments[layer_index][j] = (self.beta_2 * self.second_moments[layer_index][j]) + \
                                                  ((1 - self.beta_2) * np.power(gradients[j], 2))

        # Make estimates unbiased
        unbiased_estimates = []
        for j in range(len(gradients)):
            m_hat = self.first_moments[layer_index][j] / (1 - self.beta_1 ** self.t)
            v_hat = self.second_moments[layer_index][j] / (1 - self.beta_2 ** self.t)
            unbiased_estimates.append((m_hat, v_hat))

        # Finally update parameters
        for j, estimates in enumerate(unbiased_estimates):
            parameters[j] -= lr * (estimates[0] / np.sqrt(estimates[1] + self.epsilon))

    def optimize(self, delta: np.ndarray, layers: List[BaseModule], lr: float, reg: float):
        for i in range(len(layers) - 1, 0, -1):
            if isinstance(layers[i], ParameterModule):
                self.perform_parameter_updates(
                    layer=layers[i],
                    layer_index=i,
                    delta=delta,
                    lr=lr,
                    reg=reg
                )

            # If differentiable update delta
            if isinstance(layers[i], DifferentiableModule):
                delta = layers[i].update_delta(delta)

        if isinstance(layers[0], ParameterModule):
            self.perform_parameter_updates(
                layer=layers[0],
                layer_index=0,
                delta=delta,
                lr=lr,
                reg=reg
            )

        # Update 'self.t'
        self.t += 1


class Trainer(object):
    '''Base class for a trainer object'''
    def __init__(self):
        pass

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validate: np.ndarray, y_validate: np.ndarray,
              layers: List[BaseModule], optimizer: Optimizer, forward_function: Callable, objective_function: ObjectiveFunction, lr: float, reg: float) -> Tuple[float, float]:
        pass


class FullGDTrainer(Trainer):
    '''Performs full gradient decsent, no batches'''
    def __init__(self):
        super().__init__()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validate: np.ndarray, y_validate: np.ndarray,
              layers: List[BaseModule], optimizer: Optimizer, forward_function: Callable, objective_function: ObjectiveFunction, lr: float, reg: float) -> Tuple[float, float]:
        # Forward pass
        training_activations = forward_function(x_train, is_training=True)
        validation_activations = forward_function(x_validate, is_training=False)

        # Compute loss
        training_loss = objective_function.loss(training_activations, y_train)
        validation_loss = objective_function.loss(validation_activations, y_validate)

        # Backpropagation
        delta = objective_function.get_delta(training_activations, y_train)
        optimizer.optimize(
            delta=delta,
            layers=layers,
            lr=lr,
            reg=reg
        )

        return training_loss, validation_loss


class MiniBatchTrainer(Trainer):
    '''Class for mini-batch training'''
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validate: np.ndarray, y_validate: np.ndarray,
              layers: List[BaseModule], optimizer: Optimizer, forward_function: Callable, objective_function: ObjectiveFunction, lr: float, reg: float) -> Tuple[float, float]:
        # Shuffle data first
        x_train, y_train = shuffle(x_train, y_train)

        # Get 'size' and 'n_batches first
        size = len(x_train)
        n_batches = int(np.ceil(size / self.batch_size))
        total_training_loss = 0

        for i in range(n_batches):
            # Get batches
            start = i * self.batch_size
            stop = min((i + 1) * self.batch_size, size)
            x_batch = x_train[start:stop, :]
            y_batch = y_train[start:stop]

            # Forward pass, and compute batch loss
            batch_activations = forward_function(x_batch, is_training=True)
            batch_loss = objective_function.loss(batch_activations, y_batch)
            total_training_loss += batch_loss

            # Get delta and perform backpropagation
            delta = objective_function.get_delta(batch_activations, y_batch)
            optimizer.optimize(
                delta=delta,
                layers=layers,
                lr=lr,
                reg=reg
            )

        # Compute validation loss
        validation_activations = forward_function(x_validate, is_training=False)
        validation_loss = objective_function.loss(validation_activations, y_validate)

        return total_training_loss / n_batches, validation_loss


class NeuralNetwork(object):
    '''Base class for a simple feed forward neural net'''
    def __init__(self, layers: List[BaseModule]):
        self.layers = layers

    def set_up(self, optimizer: Optimizer, objective_function: ObjectiveFunction, batch_size=None):
        print("Setting up neural network...")
        self.objective_function = objective_function
        optimizer.set_up(self.layers)
        if batch_size is None:
            self.trainer = FullGDTrainer()
        elif isinstance(batch_size, int) and batch_size > 0:
            self.trainer = MiniBatchTrainer(batch_size=batch_size)
        else:
            raise Exception("Invalid batch size, please enter a positive integer batchsize or leave as None")
        print("Set up complete")

    def forward(self, forward_input: np.ndarray, is_training=True) -> np.ndarray:
        out = forward_input
        for layer in self.layers:
            out = layer.forward(out, is_training)
        return out

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, objective_function: ObjectiveFunction, optimizer: Optimizer,
            lr=10e-6, epochs=10000, reg=0.0, validation_prop=0.33, validation_data=None, batch_size=None, show_fig=False):
        # Set up
        self.set_up(optimizer, objective_function, batch_size)

        if validation_data is None:
            # Shuffle data
            x_train, y_train = shuffle(x_train, y_train)
            # Split data into train and test sets
            validation_index = int(len(x_train) * validation_prop)
            x_train, x_validate = x_train[:-validation_index, :], x_train[-validation_index:, :]
            y_train, y_validate = y_train[:-validation_index], y_train[-validation_index:]
        else:
            x_validate, y_validate = validation_data[0], validation_data[1]

        # Save these
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):

            # Compute forward pass and perform gradient descent, obtain training loss and validation loss
            training_loss, validation_loss = self.trainer.train(
                x_train=x_train,
                y_train=y_train,
                x_validate=x_validate,
                y_validate=y_validate,
                layers=self.layers,
                optimizer=optimizer,
                forward_function=self.forward,
                objective_function=self.objective_function,
                lr=lr,
                reg=reg
            )

            # Save these
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            # Display every 'n' epochs
            if epoch % 1 == 0:
                print(f'epoch {epoch}')
                print(f'Training Loss: {training_loss:.4f}')
                print(f'Validation Loss: {validation_loss:.4f}')

        # Display loss
        if show_fig:
            plt.title('Loss Per Epoch')
            plt.plot(validation_losses, color='red', label='Validation Loss Per Epoch')
            plt.plot(training_losses, color='blue', label='Training Loss Per Epoch')
            plt.legend()
            plt.show()

    def predict(self, x: np.ndarray) -> np.ndarray:
        activations = self.forward(x, is_training=False)
        return self.objective_function.predict(activations)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(x)
        return ut.classification_rate(y, predictions)


def main():
    x_train, x_validate, y_train, y_validate = ut.get_mnist()
    k_classes = len(set(y_train))
    ann = NeuralNetwork(
        layers=[
            LinearLayer(n_in=x_train.shape[1], n_out=1200, bias=True),
            BatchNormalization(size=1200),
            ReLU(),
            LinearLayer(n_in=1200, n_out=600, bias=True),
            BatchNormalization(size=600),
            ReLU(),
            LinearLayer(n_in=600, n_out=300, bias=True),
            BatchNormalization(size=300),
            ReLU(),
            LinearLayer(n_in=300, n_out=k_classes),
            Softmax()
        ]
    )

    # scheduler = RMSProp()
    # optimizer = StandardOptimizer(lr_scheduler=scheduler, momentum="nesterov", mu=0.95)
    optimizer = AdamOptimizer()
    objective_func = SparseCategoricalCrossEntropy()
    ann.fit(x_train, y_train, objective_function=objective_func, optimizer=optimizer, lr=10e-3, batch_size=50, epochs=10, reg=0.,
            validation_data=(x_validate, y_validate), show_fig=True)
    final_training_classification_rate = ann.score(x_train, y_train)
    final_validation_classification_rate = ann.score(x_validate, y_validate)
    print(f"Final training classification rate: {final_training_classification_rate:.4f}")
    print(f"Final validation classification rate: {final_validation_classification_rate:.4f}")


if __name__ == '__main__':
    main()





















