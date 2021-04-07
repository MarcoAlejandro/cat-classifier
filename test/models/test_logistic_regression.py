from cat_classifier.models.logistic_regression import (
    sigmoid,
    get_zero_parameters,
    forward_backward_propagate,
    optimize,
    predict,
)
import numpy as np


def test_sigmoid():
    expected_array = np.array([0.5, 0.88079708])
    assert sigmoid(np.array([0, 2])).size == expected_array.size
    assert np.allclose(sigmoid(np.array([0, 2])), expected_array)


def test_zero_parameters():
    w, b = get_zero_parameters(100)
    assert w.shape == (100, 1)
    assert b == 0


def test_forward_and_backward_propagation():
    w, b, X, Y = np.array([[1.], [2.]]), \
                 2., \
                 np.array([[1., 2., -1.], [3., 4., -3.2]]), \
                 np.array([[1, 0, 1]])

    cost, dw, db = forward_backward_propagate(w, b, X, Y)

    expected_dw = np.array([
        [0.99845601],
        [2.39507239]
    ])
    expected_db = 0.00145557813678

    assert np.allclose(expected_dw, dw)
    assert np.allclose(db, expected_db)
    assert np.allclose(cost, 5.80154531939)


def test_optimize():
    w, b, X, Y = np.array([[1.], [2.]]), \
                 2., \
                 np.array([[1., 2., -1.], [3., 4., -3.2]]), \
                 np.array([[1, 0, 1]])

    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

    expected_w = np.array([[ 0.19033591], [ 0.12259159]])
    expected_b = 1.92535983008
    expected_dw = np.array([[0.67752042], [1.41625495]])
    expected_db = 0.219194504541

    assert np.allclose(params["w"], expected_w)
    assert np.allclose(params["b"], expected_b)
    assert np.allclose(grads["dw"], expected_dw)
    assert np.allclose(grads["db"], expected_db)


def test_predict():
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    prediction = predict(w, b, X)
    expected = np.array([1.0, 1.0, 0.0])
    assert np.allclose(prediction, expected)
