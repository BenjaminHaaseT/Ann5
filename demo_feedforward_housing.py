import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nnlayers
import torch
import torch.nn as nn
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# Load data
data, targets = load_boston(return_X_y=True)

# Scale data
scalar = StandardScaler()
data = scalar.fit_transform(data)

# Convert to torch tensors
x_torch = torch.from_numpy(data.astype(np.float32))
y_torch = torch.from_numpy(targets.astype(np.float32).reshape((len(targets), 1)))

# Create model
model = nn.Sequential(
    nn.Linear(x_torch.shape[1], 300),
    nn.ReLU(),
    nn.Linear(300, 150),
    nn.ReLU(),
    nn.Linear(150, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Save loss
total_loss = []

for epoch in range(2000):
    # Zero grad
    optimizer.zero_grad()

    # Forward pass
    activations = model(x_torch)
    loss = criterion(activations, y_torch)
    total_loss.append(loss.item())

    # Gradient descent
    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print(f"Loss: {loss.item():.4f}")


# Plot loss per epoch
plt.title("Loss Per Epoch")
plt.plot(total_loss)
plt.show()

# Test vs. ann5
model = nnlayers.NeuralNetwork(
    layers=[
        nnlayers.LinearLayer(n_in=data.shape[1], n_out=300),
        nnlayers.ReLU(),
        nnlayers.LinearLayer(n_in=300, n_out=150),
        nnlayers.ReLU(),
        nnlayers.LinearLayer(n_in=150, n_out=50),
        nnlayers.ReLU(),
        nnlayers.LinearLayer(50, 1)
    ],
    objective=nnlayers.MeanSquaredError()
)

# Use same train data as test data for conveneince
x_train, y_train = data, targets
x_test, y_test = data, targets

optimizer = nnlayers.AdamOptimizer()

model.fit(
    x_train=x_train,
    y_train=y_train,
    optimizer=optimizer,
    lr=.1,
    epochs=2000,
    reg=0.,
    validation_data=(x_test, y_test),
    show_fig=True
)






