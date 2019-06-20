import numpy as np
import matplotlib.pyplot as plt
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig.reshape(-1), test_set_x_orig, test_set_y_orig.reshape(-1), classes


def generate_normal(n_samples, p=0.8, seed=2019):
    np.random.seed(seed)
    n = n_samples // 2
    n_train = int(n * p)
    X1 = np.random.normal(loc=10, scale=5, size=(n, 2))
    X2 = np.random.normal(loc=20, scale=5, size=(n, 2))
    Y1 = np.ones(n)
    Y2 = np.zeros(n)
    X_train = np.concatenate((X1[:n_train], X2[:n_train]))
    X_test = np.concatenate((X1[n_train:], X2[n_train:]))
    Y_train = np.concatenate((Y1[:n_train], Y2[:n_train]))
    Y_test = np.concatenate((Y1[n_train:], Y2[n_train:]))
    return (X_train.T, Y_train), (X_test.T, Y_test)

def plot(data, labels, title='Train data'):
    plt.scatter(data.T[labels==1][:, 0], data.T[labels==1][:, 1], color='b', edgecolor='k', label='label : 1')
    plt.scatter(data.T[labels==0][:, 0], data.T[labels==0][:, 1], color='r', edgecolor='k', label='label : 0')
    plt.grid(True)
    plt.title(title)
    plt.legend()
    
    
def decision_boundary(w, b, xlim, ylim, colormap):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    grids = np.c_[xx.ravel(), yy.ravel()]
    predict = forward(w, b, grids.T)
    Z = predict.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=0.5, colors='k')
    if colormap == True:
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.7)
        
        
def draw_boundary(w, b, data, labels, title='Train data', colormap=False):
    # 먼저 데이터 플롯한다
    plot(data, labels, title=title)
    axes = plt.gca() # 현재 플롯된 axes객체를 가져온다
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # 학습모델의 Decision boundary
    decision_boundary(w, b, xlim, ylim, colormap)
    
# 데이터의 평균과 표준편차가 각각 0과 1이 되도록 정규화하는 함수
def Normalize(data):
    return (data - data.mean()) / data.std()

def forward(w, b, X):
    m = X.shape[1]
    Yhat = sigmoid(np.dot(w.T, X) + b)
    return Yhat

def sigmoid(z):
    return 1 / (1 + np.exp(-z))