import matplotlib.pyplot as plt
import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    A = np.maximum(0, Z)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


def initialize_parameters_deep(layer_dims):
    L = len(layer_dims)
    parameters = {}
    np.random.seed(1)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / (np.sqrt(layer_dims[l - 1]))
        # parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.001
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        A, cache = linear_activation_forward(A_prev, W, b, activation="relu")
        caches.append(cache)

    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation="sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    dA_prev, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                               activation="relu")

        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def predict(X, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))

    probs, cache = L_model_forward(X, parameters)
    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


def accuracy(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))

    probs, cache = L_model_forward(X, parameters)
    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    accuracy = np.sum((p == Y) / m)

    return accuracy


def print_mislabelled_images(classes, X, Y, p):
    a = p + Y
    mislabelled_indices = np.asarray(np.where(a == 1))

    plt.rcParams['figure.figsize'] = (40.0, 40.0)
    num_images = len(mislabelled_indices[0])
    for i in range(num_images):
        index = mislabelled_indices[1][i]

        plt.subplot(1, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis = 'off'
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[Y[0, index]].decode(
                "utf-8"))
