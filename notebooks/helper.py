import numpy as np
import copy


def test1():
    print("test 1")

def test():
    test1()
    print("Hello, World!")

# Manually doing dot product
def predict(feature, parameter, intercept):
    prediction = 0
    n = feature.shape[0]
    for i in range(n):
        prediction += feature[i] * parameter[i]
    prediction += intercept
    return prediction

# Dot product using numpy
def predict_numpy(feature, parameter, intercept):
    prediction = np.dot(feature, parameter) + intercept
    return prediction

# Mean Squred Error calculation
def compute_cost(x_test, y_test, w, b):
    n = x_test.shape[0]
    cost = 0.0
    for i in range(n):
        f_wb_i = np.dot(x_test[i], y_test[i]) + b
        cost = (f_wb_i - y_test[i]) ** 2
    cost = cost / (2 * n)
    return cost

# Gradien Descent calculation
# Baiscally it adjusts the parameter and intercept to get the best results from model
def compute_gradient(x_train, y_train, w, b):
    m, n = x_train.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x_train[i], w) + b) - y_train[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x_train[i, j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


# Method to calculate the Gradient descent
def gradient_descent(x, y, weight, bias, alpha, iterations):
    # data
    # calculate the error
    cost_history = []
    w = copy.deepcopy(weight)
    b = bias
    print(type(b))
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(x, y, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        cost_history.append(compute_cost(x, y, w, b))
    print(type(b))
    return w, b, cost_history
