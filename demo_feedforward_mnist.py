import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nnlayers
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


train_data = torchvision.datasets.MNIST(
    root=".",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.MNIST(
    root=".",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create model, instantiate criterion and optimizer
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)

# Pass model to device
model.to(device)

# Save loss per iteration
training_losses = []
test_losses = []

# Train model
for epoch in range(10):
    tot_train_loss, n_batches = 0, 0
    for inputs, targets in train_loader:
        # Zero grad
        optimizer.zero_grad()

        # Send inputs, targets to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Reshape inputs
        inputs = inputs.view(-1, 784)

        # Forward pass
        activations = model(inputs)
        curr_train_loss = criterion(activations, targets)
        tot_train_loss += curr_train_loss.item()
        n_batches += 1

        # Gradient descent
        curr_train_loss.backward()
        optimizer.step()

    # Compute mean to total training loss
    train_loss = tot_train_loss / n_batches
    training_losses.append(train_loss)

    # Repeat for test data
    tot_test_loss, n_batches = 0, 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(-1, 784)
        activations = model(inputs)
        curr_test_loss = criterion(activations, targets)
        tot_test_loss += curr_test_loss.item()
        n_batches += 1

    # Compute mean of total testing loss
    test_loss = tot_test_loss / n_batches
    test_losses.append(test_loss)

    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


# Plot loss per epoch
plt.title("Loss Per Epoch")
plt.plot(training_losses, color="blue", label="Training Loss")
plt.plot(test_losses, color="red", label="Test Loss")
plt.legend()
plt.show()

# Get accuracy
with torch.no_grad():
    n_correct, n_total = 0, 0
    for inputs, targets in train_loader:
        # move to device
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(-1, 784)
        activations = model(inputs)
        _, predictions = torch.max(activations, 1)
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    training_acc = n_correct / n_total

    # Repeat for test set
    n_correct, n_total = 0, 0

    for inputs, targets in test_loader:
        # Move to device
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(-1, 784)
        activations = model(inputs)
        _, predictions = torch.max(activations, 1)
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total


print(f"Final training accuracy: {training_acc:.4f}, Final Testing accuracy: {test_acc:.4f}")


# Trial with nnlayers.py
x_train, y_train = train_data.data, train_data.targets
x_test, y_test = test_data.data, test_data.targets

x_train, y_train = x_train.numpy(), y_train.numpy()
x_test, y_test = x_test.numpy(), y_test.numpy()

train = []
test = []
for i in range(x_train.shape[0]):
    train.append(x_train[i].flatten())
for i in range(x_test.shape[0]):
    test.append(x_test[i].flatten())

x_train = np.array(train).astype(np.float32)
x_train /= 255.

x_test = np.array(test).astype(np.float32)
x_test /= 255.

model = nnlayers.NeuralNetwork(
    layers=[
        nnlayers.LinearLayer(x_train.shape[1], 128),
        nnlayers.ReLU(),
        nnlayers.LinearLayer(128, 10),
        nnlayers.Softmax()
    ],
    objective=nnlayers.SparseCategoricalCrossEntropy()
)

optimizer = nnlayers.AdamOptimizer(model.layers)

model.fit(
    x_train=x_train,
    y_train=y_train,
    optimizer=optimizer,
    lr=10e-4,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test),
    show_fig=True
)

train_acc = model.score(x_train, y_train)
valid_acc = model.score(x_test, y_test)
print(f"Final training accuracy: {train_acc:.4f}, final testing accuracy: {valid_acc:.4f}")


# Regression
# Generate data
N = 10000
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2 * X[:, 0]) + np.cos(3 * X[:, 1])
Y = Y.reshape(len(Y), 1)

# Plot data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))

# Build model
model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(1000):
    # zero grad
    optimizer.zero_grad()

    # Forward pass
    activations = model(X)
    loss = criterion(activations, Y)
    losses.append(loss.item())

    # Gradient descent
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/1000, Loss: {loss.item():.4f}")

# Plot loss per epoch
plt.title("Loss Per Epoch")
plt.plot(losses)
plt.show()

# Plot predicted surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot
with torch.no_grad():
    line = np.linspace(-3, 3, 50)
    x1, x2 = np.meshgrid(line, line)
    x_grid = np.vstack((x1.flatten(), x2.flatten())).T
    x_grid_torch = torch.from_numpy(x_grid.astype(np.float32))
    y_hat = model(x_grid_torch).numpy().flatten()
    ax.plot_trisurf(x_grid[:, 0], x_grid[:, 1], y_hat)
    plt.show()

# Test with ann5
model = nnlayers.NeuralNetwork(
    layers=[
        nnlayers.LinearLayer(n_in=2, n_out=128),
        nnlayers.ReLU(),
        nnlayers.LinearLayer(128, 1)
    ],
    objective=nnlayers.MeanSquaredError()
)
