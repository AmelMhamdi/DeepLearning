import numpy as np
import matplotlib.pyplot as plt


def init_variables():
    """
    Init model variables (weights, bias)
    :return: initialiser le poid du chaque feature
    """
    weights = np.random.normal(size=2)
    bias = 0
    return weights, bias


def get_dataset():
    """
    Methode used to generate a dataset
    :return: features et targets
    """
    # Number of row per class
    row_per_class = 5
    # Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])

    features = np.vstack([sick, healthy])
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    return features, targets


def pre_activation(features, weights, bias):
    """
    compute pre activation
    :param features: Input
    :param weights: poid du chaque feature
    :param bias: bias
    :return: valeur de pre activation
    """

    return np.dot(features, weights) + bias


def activation(z):
    """
    compute activation
    :param z: pre activation
    :return: valeur d'activation
    """
    return 1 / (1 + np.exp(-z))


def derivative_activation(z):
    """

    :param z:
    :return:
    """
    return activation(z) * (1 - activation(z))


def cost(predictions, targets):
    """"
    """
    return np.mean((predictions - targets) ** 2)


def predict(features, weights, bias):
    """

    :param features:
    :param weights:
    :param bias:
    :return:
    """
    z = pre_activation(features, weights, bias)
    y = activation(z)
    return np.round(y)


def train(features, targets, weights, bias):
    """

    :param features:
    :param targets:
    :param weights:
    :param bias:
    :return:
    """
    epochs = 100
    learning_rate = 0.1

    # Print current Accuracy
    predictions = predict(features, weights, bias)
    print("Accuracy", np.mean(predictions == targets))

    # Plot points
    # plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    # plt.show()

    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print(predictions)
            print("cost = %s" % cost(predictions, targets))
            # Initiat gradients
            weights_gradients = np.zeros(weights.shape)
            bias_gradient = 0
            # Go through each row
            for feature, target in zip(features, targets):
                # Compute prediction
                z = pre_activation(feature, weights, bias)
                y = activation(z)
                # Update gradients
                weights_gradients += (y - target) * derivative_activation(z) * feature
                bias_gradient += (y - target) * derivative_activation(z)
            # update variables
            weights = weights - learning_rate * weights_gradients
            bias = bias - learning_rate * bias_gradient
            predictions = predict(features, weights, bias)
            print("Accuracy", np.mean(predictions == targets))


if __name__ == '__main__':
    # Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    z = pre_activation(features, weights, bias)
    a = activation(z)
    train(features, targets, weights, bias)
    print(targets)
    print(a)
    pass
