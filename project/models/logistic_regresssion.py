import numpy as np


def sigmoid(z):
    """
        Computes the sigmoid function of z.

        Args:
            z: an scalar or a numpy array of any size.

        Returns:
            Sigmoid of z.
    """
    s = 1 / (1 + np.exp(-z))
    return s


def get_zero_parameters(dim):
    """
        returns an tuple where first element is an array with (dim,1) dimensions,
        with all values set to zero for the weights,
        and the second element is the b scalar bias.
    """
    return np.zeros((dim, 1)), 0


def forward_backward_propagate(
        w: np.ndarray,
        b: float,
        X: np.ndarray,
        Y: np.ndarray
):
    """
        Performs a forward propagation to compute the cost function on the input
        features vector X, and the labels Y.

        Args:
            w: weights for the features.
            b: threshold
            X: input vectors
            Y: True/false (1/0) labels.

        Returns:
            Returns a three elements tuple where:
                first element is the 'cost' for logistic regression
                second element is 'dw', the gradient of the loss with respect to w, thus same shape as w
                third element 'db' is the gradient of the loss with respect to b, thus same shape as b
    """
    # m is the number of training samples
    m = X.shape[1]

    # Computes the value of the activation function each one of the input vectors.
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/m) * (np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))

    # Given the cost, is possible to perform the gradient descent
    dZ = A - Y
    dw = 1/m * np.dot(X, dZ.T)
    db = 1/m * np.sum(dZ)

    return cost, dw, db


def optimize(
        w: np.array,
        b,
        X,
        Y,
        num_iterations,
        learning_rate,
        print_cost=False
):
    """
    Performs certain number of gradient descent updates, each loop optimizes w and b.

    Args:
        w: weights for features
        b: threshold value
        X: array of features vectors
        Y: labels for the vectors
        num_iterations: number of iterations of gradient descents
        learning_rate: learning rate of the gradient descent update rule
        print_cost: True to print the loss every 100 steps
    """

    costs = []
    for i in range(num_iterations):
        cost, dw, db = forward_backward_propagate(
            w,
            b,
            X,
            Y
        )
        costs += cost
        if cost % 100 == 0 and print_cost:
            print(f"Cost for iteration {i}: {cost}")

        w = w - learning_rate * dw
        b = b - learning_rate * db

    params = {
        "w": w,
        "b": b
    }

    gradients = {
        "dw": dw,
        "db": db
    }

    return params, gradients, costs


def predict(
        w,
        b,
        X
):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b),
    for the input vectors in X.

    Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X

    """
    # init parameters
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0

    return Y_prediction

