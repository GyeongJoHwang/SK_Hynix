import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # generating synthetic data


def generate_unfair_data(n_samples, p=0.8, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
    # this determines the discrimination in the data - decraese it to generate more discrimination
    # disc_factor = math.pi / 4.0 (default)
    disc_factor = math.pi / 8.0
    n = n_samples // 2
    num_train = int(n_samples * p)

    def generate_gaussian(mean, cov, label):
        pdf = multivariate_normal(mean=mean, cov=cov)
        X = pdf.rvs(n)
        Y = np.ones(n, dtype=float) * label
        return pdf, X, Y

    """Generate the non-sensitive features randomly"""
    # We will generate one gaussian cluster for each class
    mean1, sigma1 = [10, 10], [[25, 0], [0, 25]]
    mean2, sigma2 = [20, 20], [[25, 0], [0, 25]]
    pdf1, X1, Y1 = generate_gaussian(mean1, sigma1, 0) # positive class
    pdf2, X2, Y2 = generate_gaussian(mean2, sigma2, 1) # negative class

    # join the posisitve and negative class clusters
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2))

    # shuffle the data
    perm = list(range(0, n_samples))
    random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]
    
    rotation_matrix = np.array([[math.cos(disc_factor), -math.sin(disc_factor)],
                                [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_matrix)

    """Generate the sensitive feature here"""
    z_list = [] # this array holds the sensitive feature value
    for i in range (0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = pdf1.pdf(x)
        p2 = pdf2.pdf(x)
        
        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s
        
        r = np.random.uniform() # generate a random number from 0 to 1

        if r < p1: # the first cluster is the positive class
            z_list.append(1.0) # 1.0 means its male
        else:
            z_list.append(0.0) # 0.0 -> female

    z = np.array(z_list)
    
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    z_train = z[:num_train]
    X_test = X[num_train:]
    Y_test = Y[num_train:]
    z_test = z[num_train:]
    return (X_train.T, Y_train, z_train), (X_test.T, Y_test, z_test)

def plot_unfair_data(X, Y, z, num_to_draw=200):
    X_draw = X[:num_to_draw]
    Y_draw = Y[:num_to_draw]
    z_draw = z[:num_to_draw]

    X_z_0 = X_draw[z_draw == 0.0]
    X_z_1 = X_draw[z_draw == 1.0]
    Y_z_0 = Y_draw[z_draw == 0.0]
    Y_z_1 = Y_draw[z_draw == 1.0]
    plt.scatter(X_z_0[Y_z_0==1.0][:, 0], X_z_0[Y_z_0==1.0][:, 1], color='red', marker='o', facecolors='none', s=90, linewidth=1.5, label="White reoffend")
    plt.scatter(X_z_0[Y_z_0==0.0][:, 0], X_z_0[Y_z_0==0.0][:, 1], color='blue', marker='o', facecolors='none', s=90, linewidth=1.5, label="White non-reoffend")
    plt.scatter(X_z_1[Y_z_1==1.0][:, 0], X_z_1[Y_z_1==1.0][:, 1], color='red', marker='o', facecolors='black', s=90, linewidth=1.5, label="Black reoffend")
    plt.scatter(X_z_1[Y_z_1==0.0][:, 0], X_z_1[Y_z_1==0.0][:, 1], color='blue', marker='o', facecolors='black', s=90, linewidth=1.5, label="Black non-reoffend")

    plt.tick_params(axis='x', which='both')#, bottom=False, top=False)#, labelbottom='off') # dont need the ticks to see the data distribution
    plt.tick_params(axis='y', which='both')#, left=False, right=False)#, labelleft='off')
    plt.legend()
    plt.show()
    return None