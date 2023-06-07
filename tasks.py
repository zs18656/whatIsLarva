import numpy as np
import pandas as pd # only used to read the MNIST data set
import networkx as nx
from datetime import datetime
import sklearn.tree
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from bter import BTER
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
import random
from time import time
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import concurrent.futures as cf
import imageio
import torch
from sklearn.inspection import permutation_importance

from sklearn.svm import SVC, LinearSVC

import os
os.path.abspath(os.getcwd())

"""
Functions here should return 5 arrays (or tensors)
ts, X_train, Y_train, X_test, Y_test

Generally, as we're using reservoirs with warmup, X_test and Y_test are continuations of the series from X_train, Y_train
"""

def frequency_prediction_data(steps = 1200, train_ratio = 0.75, n_cycles = 3, noise_amount = 0.):
    n_in_train = int(train_ratio * steps)
    ts = np.linspace(0, n_cycles * np.pi, num=steps)

    t_noise = np.random.randn(steps)
    # Add more noise in non-train section
    t_noise *= noise_amount  # 05
    # t_noise[n_in_train:] *= 0.05

    frequencies = np.cos(ts)
    amp_saved = np.sin(np.pi * frequencies) + t_noise
    amplitudes = np.copy(amp_saved).reshape((1, -1))
    amplitudes = np.concatenate((amplitudes, amplitudes), axis=0)
    # print(frequencies.shape)

    frequencies_test = frequencies
    frequencies = frequencies[:n_in_train]
    amplitudes_test = amplitudes
    amplitudes = amplitudes[:, :n_in_train]

    return ts, amplitudes, frequencies, amplitudes_test, frequencies_test

def autoregression_data(steps = 400, train_ratio = 0.25, total_range = 16*np.pi, noise_amount = 0.1, function = np.sin):
    n_in_train = int(train_ratio*steps)
    ts = np.linspace(0, total_range, num=steps + 1) # +1 for autoregression

    noise = np.random.randn(steps + 1)
    # Add more noise in non-train section
    noise *= noise_amount  # 05
    # t_noise[n_in_train:] *= 0.05

    all_function_values = function(ts)  + noise

    X = all_function_values[:-1]
    Y = all_function_values[1:]

    assert X.shape == Y.shape, "Something has gone wrong in data generation, X and Y should be same shape here"
    print(X)
    X = np.expand_dims(X, axis = 0)
    print(X)

    X_train, Y_train = X[:, :n_in_train], Y[:n_in_train]

    return ts[1:], X_train, Y_train, X, Y

# def

