# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def generate_normal(n_samples, p=0.8, seed=20):
    np.random.seed(seed)
    
    n = n_samples // 2
    n_train = int(n * p)
    n_val = int(n * 0.08)
    
    X1 = np.random.normal(loc=10, scale=5, size=(n, 2))
    X2 = np.random.normal(loc=20, scale=5, size=(n, 2))
    Y1 = np.zeros(n)
    Y2 = np.ones(n)
    
    X_train = np.concatenate((X1[:n_train - n_val], X2[:n_train - n_val]))
    Y_train = np.concatenate((Y1[:n_train - n_val], Y2[:n_train - n_val]))
    X_val = np.concatenate((X1[n_train - n_val : n_train], X2[n_train - n_val : n_train]))
    Y_val = np.concatenate((Y1[n_train - n_val : n_train], Y2[n_train - n_val : n_train]))
    X_test = np.concatenate((X1[n_train:], X2[n_train:]))
    Y_test = np.concatenate((Y1[n_train:], Y2[n_train:]))
    
    return (X_train.T, Y_train), (X_val.T, Y_val), (X_test.T, Y_test)

def plot(data, labels, title='Train data', s=35, axis=False, xlim=None, ylim=None):
    plt.scatter(data.T[labels==1][:, 0], data.T[labels==1][:, 1], color='b', edgecolor='k', label='label : 1', s=s)
    plt.scatter(data.T[labels==0][:, 0], data.T[labels==0][:, 1], color='r', edgecolor='k', label='label : 0', s=s)
    plt.grid(True)
    plt.title(title)
    plt.legend()
    if axis:
        plt.axvline(x=0, color='black', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
        
def decision_boundary(w, b, xlim, ylim, colormap):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    grids = np.c_[xx.ravel(), yy.ravel()]
    predict = forward_(w, b, grids.T)
    Z = predict.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k')
    if colormap == True:
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.7)

def draw_boundary(w, b, data, labels, title='Train data', colormap=False, s=35, axis=False, xlim=None, ylim=None):
    # 먼저 데이터 플롯한다
    plot(data, labels, title=title, s=s, axis=axis, xlim=xlim, ylim=ylim)
    axes = plt.gca() # 현재 플롯된 axes객체를 가져온다
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # 학습모델의 Decision boundary
    decision_boundary(w, b, xlim, ylim, colormap)

def check_PSD(M):
    return np.all(np.linalg.eigvals(M) >= 0)

# 데이터의 평균과 표준편차가 각각 0과 1이 되도록 정규화하는 함수
def Normalize(data):
    return (data - data.mean()) / data.std()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

def forward_(w, b, X):
    m = X.shape[1]
    Yhat = sigmoid(np.dot(w.T, X) + b)
    return Yhat

def forward(w, b, X, Y):
    m = X.shape[1]
    Yhat = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(Yhat) + (1 - Y) * (np.log(1 - Yhat)))
    return Yhat, cost

def backward(w, b, X, Y, Yhat):
    m = X.shape[1]
    dw = (1 / m) * np.dot(X, (Yhat - Y).T)
    db = (1 / m) * np.sum(Yhat - Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    grads = {"dw": dw, "db": db}
    return grads

def fit(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1-4 lines of code)
        Yhat, cost = forward(w, b, X, Y)
        grads = backward(w, b, X, Y, Yhat)
        
        # Retrieve derivatives from grads
        dw, db = grads["dw"], grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and (i + 1) % 100 == 0 or (i + 1) == num_iterations:
            print ("Cost after iteration [%i/%i]: %f" % (i + 1, num_iterations, cost))#, end='\r')
    print()
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    assert(w.shape[0] == X.shape[0])
    
    Yhat = sigmoid(np.dot(w.T, X) + b)
    Yhat[Yhat > 0.5] = 1
    Yhat[Yhat <= 0.5] = 0
    Y_prediction = Yhat
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def Logistic(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = fit(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w, b = parameters["w"], parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test accuracies
    train_acc = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    print("train accuracy: {} %".format(train_acc))
    print("test accuracy: {} %".format(test_acc))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train,
         "train_acc": train_acc,
         "test_acc": test_acc,
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d