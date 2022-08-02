import nnlayers
import opt
import utils as ut
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


def main():
    # Download data as torch tensors
    train_data = torchvision.datasets.MNIST(
        root='.',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    validation_data = torchvision.datasets.MNIST(
        root='.',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    # Convert to np for Ann5
    train_data_np = train_data.data.detach().numpy()
    train_targets_np = train_data.targets.detach().numpy()
    validation_data_np = validation_data.data.detach().numpy()
    validation_targets_np = validation_data.targets.detach().numpy()
    # Reshape for cnn
    train_data_np = train_data_np[:, :, :, np.newaxis]
    validation_data_np = validation_data_np[:, :, :, np.newaxis]

    train_loader = ut.BatchLoader(data=train_data_np,
                                  targets=train_targets_np,
                                  batch_size=32,
                                  shuffle=True)

    validation_loader = ut.BatchLoader(data=validation_data_np,
                                       targets=validation_targets_np,
                                       batch_size=128,
                                       shuffle=False)

    # Build model
    model = nnlayers.Sequential(
        nnlayers.Conv2d(channels_in=train_data_np.shape[3], channels_out=32, kernel_size=(3, 3), padding=1, stride=2),
        nnlayers.ReLU(),
        nnlayers.Conv2d(channels_in=32, channels_out=64, kernel_size=(3, 3), padding=1, stride=2),
        nnlayers.ReLU(),
        nnlayers.Conv2d(channels_in=64, channels_out=128, kernel_size=(3, 3), padding=1, stride=2),
        nnlayers.ReLU(),
        nnlayers.Conv2d(channels_in=128, channels_out=256, kernel_size=(3, 3), padding=1, stride=2),
        nnlayers.ReLU(),
        nnlayers.Flatten(),
        nnlayers.LinearLayer(2 * 2 * 256, 2048),
        nnlayers.ReLU(),
        nnlayers.LinearLayer(2048, 10),
        nnlayers.Softmax()
    )
    # model = nnlayers.Sequential(
    #     layers=[
    #         nnlayers.Conv2d(channels_in=train_data_np.shape[3], channels_out=16, kernel_size=(3, 3), padding=1),
    #         nnlayers.ReLU(),
    #         nnlayers.MaxPooling(filter_size=(2, 2)),
    #         nnlayers.Conv2d(channels_in=16, channels_out=32, kernel_size=(3, 3), padding=1),
    #         nnlayers.ReLU(),
    #         nnlayers.MaxPooling(filter_size=(2, 2)),
    #         nnlayers.Conv2d(channels_in=32, channels_out=64, kernel_size=(3, 3), padding=1),
    #         nnlayers.ReLU(),
    #         nnlayers.MaxPooling(filter_size=(2, 2)),
    #         nnlayers.Conv2d(channels_in=64, channels_out=128, kernel_size=(3, 3), padding=1),
    #         nnlayers.ReLU(),
    #         nnlayers.MaxPooling(filter_size=(2, 2)),
    #         nnlayers.Flatten(),
    #         nnlayers.LinearLayer(128, 2048),
    #         nnlayers.ReLU(),
    #         nnlayers.LinearLayer(2048, 10),
    #         nnlayers.Softmax()
    #     ]
    # )

    # optimizer = nnlayers.AdamOptimizer(layers=model.get_layers(), lr=0.0001)
    optimizer = opt.AdamOptimizer(layers=model.get_layers(), lr=10e-4)
    criterion = nnlayers.SparseCategoricalCrossEntropy()

    train_losses, validation_losses = [], []
    train_accuracy, validation_accuracy = [], []
    epochs = 15

    for epoch in range(epochs):
        batch_train_losses = []
        batch_train_accuracies = []
        t0 = dt.datetime.utcnow()

        for inputs, targets in train_loader:
            # Train mode
            model.train()
            t1 = dt.datetime.utcnow()
            activations = model(inputs)
            batch_loss = criterion.loss(activations, targets)
            batch_train_losses.append(batch_loss)
            batch_predictions = criterion.predict(activations)
            batch_acc = np.mean(batch_predictions == targets)
            batch_train_accuracies.append(batch_acc)
            # Gradient descent
            delta = criterion.get_delta(activations, targets)
            optimizer.optimize(delta)
            delta_time = dt.datetime.utcnow() - t1
            print(f"Time to complete batch: {delta_time}")
            print(f"Batch Loss: {batch_loss:.4f}, Batch Accuracy: {batch_acc:.4f}")

        batch_time_delta = dt.datetime.utcnow() - t0
        train_loss, train_acc = np.mean(batch_train_losses), np.mean(batch_train_accuracies)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Completed in: {batch_time_delta}")


if __name__ == "__main__":
    main()

