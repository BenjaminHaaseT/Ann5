import numpy as np
import matplotlib.pyplot as plt
import ann5
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

        # Send inputs, targets to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Reshape inputs
        inputs = inputs.view(-1, 784)

        optimizer.zero_grad()

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






# Trial with ann5.py
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

model = ann5.NeuralNetwork(
    layers=[
        ann5.LinearLayer(x_train.shape[1], 128),
        ann5.ReLU(),
        ann5.LinearLayer(128, 10),
        ann5.Softmax()
    ],
    objective=ann5.SparseCategoricalCrossEntropy()
)

optimizer = ann5.AdamOptimizer()

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
print(f"Final train acc: {train_acc:.4f}, final test acc: {valid_acc:.4f}")
