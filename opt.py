import numpy as np
from typing import List, Tuple
from nnlayers import BaseModule, ParameterModule, DifferentiableModule


MOMENTUM_UPDATERS = {
    'standard': (lambda velocity, differential, mu: velocity),
    'nesterov': (lambda velocity, differential, mu: (mu * velocity) - differential)
}


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


class CachedLearningRateScheduler(LearningRateScheduler):
    """Abstract base class for all learning rate schedulers that implement a cache for parameters"""
    def __init__(self, epsilon=10e-8):
        super().__init__()
        self.epsilon = epsilon
        self.cache = {}

    def set_up(self, layers: List[BaseModule]) -> None:
        # Set up cache with initial values
        for i in range(len(layers)):
            if isinstance(layers[i], ParameterModule):
                layer_cache = {}
                parameters = layers[i].params()
                for j in range(len(parameters)):
                    layer_cache[j] = np.ones_like(parameters[j])
                # Add layer cache to self.cache
                self.cache[i] = layer_cache


class AdaGrad(CachedLearningRateScheduler):
    '''Provides AdaGrad adaptive learning rate functionality'''
    def __init__(self, epsilon=10e-8):
        super().__init__(epsilon)

    def update_cache(self, layer_index: int, gradients: List[np.ndarray]) -> None:
        for j in range(len(gradients)):
            self.cache[layer_index][j] += np.power(gradients[j], 2)

    def get_cache_updates(self, layer_index: int, gradients: List[np.ndarray]) -> List[np.ndarray]:
        return [
            gradients[j] / np.sqrt(self.cache[layer_index][j] + self.epsilon) for j in range(len(gradients))
        ]


class RMSProp(CachedLearningRateScheduler):
    '''Provides RMS Prop learning rate scheduling'''
    def __init__(self, decay=0.999, epsilon=10e-8):
        super().__init__(epsilon)
        self.decay = decay

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
    def __init__(self, layers: List[BaseModule], lr: float = 10e-3):
        self.layers = layers
        self.lr = lr

    def set_up(self) -> None:
        pass

    def perform_parameter_updates(self, layer: ParameterModule, delta: np.ndarray, reg: float) -> None:
        pass

    def optimize(self, delta: np.ndarray, reg: float = 0) -> None:
        pass


class CachedOptimizer(Optimizer):
    '''Base class for all optimizers that need to update a `cache` of some kind during gradient descent'''
    def __init__(self, layers: List[BaseModule], lr: float = 10e-3):
        super().__init__(layers, lr)

    def perform_parameter_updates(self, layer: ParameterModule, layer_index: int, delta: np.ndarray, reg: float) -> None:
        pass

    def optimize(self, delta: np.ndarray, reg: float = 0) -> None:
        # Perform backpropagation
        for i in range(len(self.layers) - 1, 0, -1):
            # If parameter module update parameters
            if isinstance(self.layers[i], ParameterModule):
                self.perform_parameter_updates(
                    layer=self.layers[i],
                    layer_index=i,
                    delta=delta,
                    reg=reg
                )

            # If differentiable module update delta term accordingly
            if isinstance(self.layers[i], DifferentiableModule):
                delta = self.layers[i].update_delta(delta)

        # Finally update input layer if necessary
        if isinstance(self.layers[0], ParameterModule):
            self.perform_parameter_updates(
                layer=self.layers[0],
                layer_index=0,
                delta=delta,
                reg=reg
            )


class VanillaOptimizer(Optimizer):
    '''Provides standard "vannilla" gradient decent, no momentum and no adaptive learning rate functionality'''
    def __init__(self, layers: List[BaseModule], lr: float = 10e-3):
        super().__init__(layers, lr)

    def perform_parameter_updates(self, layer: ParameterModule, delta: np.ndarray, reg: float) -> None:
        '''Helper method, to eliminate repeated code'''
        gradients = layer.get_gradients(delta, reg)
        parameters = layer.params()
        for j in range(len(parameters)):
            parameters[j] -= self.lr * gradients[j]

    def optimize(self, delta: np.ndarray, reg: float = 0):
        for i in range(len(self.layers) - 1, 0, -1):
            # Update parameters if necessary
            if isinstance(self.layers[i], ParameterModule):
                self.perform_parameter_updates(self.layers[i], delta, reg)

            # If differentiable, update delta term
            if isinstance(self.layers[i], DifferentiableModule):
                delta = self.layers[i].update_delta(delta)

        # Finally update input layer if necessary
        if isinstance(self.layers[0], ParameterModule):
            self.perform_parameter_updates(self.layers[0], delta, reg)


class MomentumOptimizer(CachedOptimizer):
    '''Provides gradient descent with momentum'''
    def __init__(self, layers: List[BaseModule], lr: float = 10e-3, momentum: str = 'standard', mu: float = 0.9):
        super().__init__(layers, lr)
        self.momentum_updater = MOMENTUM_UPDATERS[momentum]
        self.mu = mu
        self.velocities = {}
        self.set_up()

    def set_up(self) -> None:
        # Set up initial velocities for each layer that has parameters i.e. 'ParameterModule' in 'layers
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], ParameterModule):
                layer_velocity = {}
                parameters = self.layers[i].params()
                for j in range(len(parameters)):
                    # Add velocity for each parameter of the layer
                    # Initialize to 0's
                    layer_velocity[j] = np.zeros_like(parameters[j])
                # Add 'layer_velocity' to self.velocities for back propagation
                self.velocities[i] = layer_velocity

    def perform_parameter_updates(self, layer: ParameterModule, layer_index: int, delta: np.ndarray, reg: float) -> None:
        '''Helper method to prevent duplicated code'''
        gradients = layer.get_gradients(delta, reg)
        parameters = layer.params()
        # Update velocity at current layer for each parameter 'j'
        for j in range(len(gradients)):
            self.velocities[layer_index][j] = (self.mu * self.velocities[layer_index][j]) - (self.lr * gradients[j])
        # Update parameters with velocity
        for j in range(len(parameters)):
            parameters[j] += self.momentum_updater(
                velocity=self.velocities[layer_index][j],
                differential=self.lr*gradients[j],
                mu=self.mu
            )


class StandardOptimizer(MomentumOptimizer):
    '''Optimizer that provides both momentum and adaptive learning rate capabilities'''
    def __init__(self, layers: List[BaseModule],
                 lr_scheduler: LearningRateScheduler,
                 lr: float = 10e-3,
                 momentum: str = "standard",
                 mu: float = 0.9):
        super().__init__(layers, lr)
        self.momentum_updater = MOMENTUM_UPDATERS[momentum]
        self.mu = mu
        self.scheduler = lr_scheduler
        self.velocities = {}
        self.set_up()

    def set_up(self) -> None:
        super().set_up()
        self.scheduler.set_up(self.layers)

    def perform_parameter_updates(self, layer: ParameterModule, layer_index: int, delta: np.ndarray, reg: float) -> None:
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
            self.velocities[layer_index][j] = (self.mu * self.velocities[layer_index][j]) - self.lr * updates[j]

        # Finally update parameters with velocity
        for j in range(len(parameters)):
            parameters[j] += self.momentum_updater(
                velocity=self.velocities[layer_index][j],
                differential=self.lr*updates[j],
                mu=self.mu
            )


class AdamOptimizer(CachedOptimizer):
    '''Provides ADAM optimization functionality'''
    def __init__(self, layers: List[BaseModule], lr: float = 10e-3, beta_1: float = 0.99, beta_2: float = 0.999, epsilon: float = 10e-8):
        super().__init__(layers, lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 1
        self.first_moments = {}
        self.second_moments = {}
        self.set_up()

    def set_up(self) -> None:
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], ParameterModule):
                # Initiate first and second moments for layer
                first_moment = {}
                second_moment = {}
                parameters = self.layers[i].params()
                for j in range(len(parameters)):
                    first_moment[j] = np.zeros_like(parameters[j])
                    second_moment[j] = np.ones_like(parameters[j])
                self.first_moments[i] = first_moment
                self.second_moments[i] = second_moment

    def perform_parameter_updates(self, layer: ParameterModule, layer_index: int, delta: np.ndarray, reg: float) -> None:
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
            parameters[j] -= self.lr * (estimates[0] / np.sqrt(estimates[1] + self.epsilon))

    def optimize(self, delta: np.ndarray, reg: float = 0):
        super().optimize(delta=delta, reg=reg)
        # Update 'self.t'
        self.t += 1