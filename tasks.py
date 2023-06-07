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
Functions here should return 4 arrays (or tensors)
X_train, Y_train, X_test, Y_test

Generally, as we're using reservoirs with warmup, X_test and Y_test are continuations of the series from X_train, Y_train
"""

def frequency_prediction(steps = 3200, train_ratio = 0.75, n_cycles = 3)