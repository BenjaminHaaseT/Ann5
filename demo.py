import ann5
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load data
data = load_breast_cancer()
X, Y = data.data, data.target
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.33)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_valid = scalar.transform(x_valid)

# Instantiate model
model = ann5.NeuralNetwork(
    layers=[
        ann5.LinearLayer(x_train.shape[1], 1),
        ann5.Sigmoid()
    ]
)

objective_function = ann5.BinaryCrossEntropy()
optimizer = ann5.AdamOptimizer()

# Fit model
model.fit(
    x_train=x_train,
    y_train=y_train,
    objective_function=objective_function,
    optimizer=optimizer,
    lr=10e-5,
    validation_data=(x_valid, y_valid),
    show_fig=True
)

training_acc = model.score(x_train, y_train)
validation_acc = model.score(x_valid, y_valid)

print(f"Final training accuracy: {training_acc:.4f}")
print(f"Final validation accuracy: {validation_acc:.4f}")



