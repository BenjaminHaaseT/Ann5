import ann5
import numpy as np
import utils as ut
import matplotlib.pyplot as plt


def main():

    x_train, x_validate, y_train, y_validate = ut.get_mnist(normalize=True)
    k_classes = len(set(y_train))
    train_loader = ut.BatchLoader(
        data=x_train,
        targets=y_train,
        batch_size=128,
        shuffle=True
    )

    test_loader = ut.BatchLoader(
        data=x_validate,
        targets=y_validate,
        batch_size=128,
        shuffle=False
    )

    model = ann5.Sequential(
        layers=[
            ann5.LinearLayer(x_train.shape[1], 512),
            ann5.ReLU(),
            ann5.LinearLayer(512, 256),
            ann5.ReLU(),
            ann5.LinearLayer(256, 128),
            ann5.ReLU(),
            ann5.LinearLayer(128, 64),
            ann5.ReLU(),
            ann5.LinearLayer(64, k_classes),
            ann5.Softmax()
        ]
    )

    model.set_up(x_train.shape)
    optimizer = ann5.AdamOptimizer(layers=model.get_layers(), lr=10e-4)
    criterion = ann5.SparseCategoricalCrossEntropy()
    train_losses, validation_losses = [], []
    train_accuracies, validation_accuracies = [], []
    epochs = 40

    for epoch in range(40):

        # Save these for finding the mean
        batch_train_losses = []
        batch_train_accuracies = []
        model.train()

        for inputs, targets in train_loader:

            # Forward pass in train mode
            activations = model(inputs)
            training_loss = criterion.loss(activations, targets)
            batch_train_losses.append(training_loss)
            predictions = np.argmax(activations, axis=1)
            training_acc = ut.classification_rate(targets, predictions.flatten())
            batch_train_accuracies.append(training_acc)

            # Gradient descent
            delta = criterion.get_delta(activations, targets)
            optimizer.optimize(delta)

        # Save these
        batch_train_loss = np.mean(batch_train_losses)
        batch_train_acc = np.mean(batch_train_accuracies)
        train_losses.append(batch_train_loss)
        train_accuracies.append(batch_train_acc)

        # Validation
        model.evaluate()
        batch_validation_losses = []
        batch_validation_accuracies = []

        for inputs, targets in test_loader:

            activations = model(inputs)
            validation_loss = criterion.loss(activations, targets)
            batch_validation_losses.append(validation_loss)
            predictions = np.argmax(activations, axis=1)
            validation_acc = ut.classification_rate(targets, predictions.flatten())
            batch_validation_accuracies.append(validation_acc)

        # Save these
        batch_validation_loss = np.mean(batch_validation_losses)
        batch_validation_accuracy = np.mean(batch_validation_accuracies)
        validation_losses.append(batch_validation_loss)
        validation_accuracies.append(batch_validation_accuracy)

        print(f"Epoch: {epoch + 1}/{epochs}, Training Loss: {batch_train_loss:.4f}, Validation Loss {batch_validation_loss:.4f}")
        print(f"Training Accuracy: {batch_train_acc:.4f}, Validation Accuracy: {batch_validation_accuracy:.4f}")

    plt.title('Loss Per Epoch')
    plt.plot(train_losses, color='blue', label='Training Loss')
    plt.plot(validation_losses, color='red', label='Validation Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()






