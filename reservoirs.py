import numpy as np
import pandas as pd # only used to read the MNIST data set
import networkx as nx
import sklearn.tree
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
import random
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.inspection import permutation_importance

from sklearn.svm import SVC, LinearSVC

import os
os.path.abspath(os.getcwd())

class Reservoir:
    def __init__(self,
                 graph,
                 prediction_model = MLPRegressor,
                 prediction_model_kwargs = {"verbose":True, "hidden_layer_sizes":(20,)}):

        self.adjacency = graph
        self.prediction_model = prediction_model(**prediction_model_kwargs)
        # Wu should be initialised when .fit is called
        self.Wx = self.create_weights(shape = tuple(self.adjacency.shape))
        self.Wu = None



    def create_weights(self, shape, low=-1.0, high=1.0, sparsity=None, spectral_radius=None):
        w = (high - low) * np.random.ranf(shape[0] * shape[1]).reshape(shape) + low  # create the weight matrix
        if not sparsity is None:  # if sparsity is defined
            s = np.random.ranf(shape[0] * shape[1]).reshape(shape) < (1.0 - sparsity)  # create a sparse boolean matrix
            w *= s  # set weight matrix values to 0.0
        if not spectral_radius is None:  # if spectral radius is defined
            sp = np.max(np.abs(np.linalg.eig(w)[0]))  # compute current spectral radius
            w *= (spectral_radius) / sp  # adjust weight matrix to acheive specified spectral radius
        return w


    def fit(self, series, targets):
        self.Wu = np.ones([self.Wx.shape[0], series.shape[0]])

        # Essentially training data, currently no warm-up
        embedded_states = self.forward(series)
        # print(embedded_states)

        self.prediction_model.fit(embedded_states, targets)
        # plot_tree(self.prediction_model)
        # plt.show()


    def predict(self, series):
        assert self.Wu is not None, "Input weight matrix doesn't exist, must call .fit before .predict"
        embedded_states = self.forward(series)

        plt.plot([np.sum(state) for state in embedded_states])
        plt.show()

        prediction = self.prediction_model.predict(embedded_states)

        return prediction


    # Alex's function for a simple reservoir operation - please sense check!
    def forward(self, input, remove_node_idxs=[]):
        # Input should be shape |V| x n_steps
        n_steps = input.shape[1]
        x_size = self.Wx.shape[0] # Network size

        assert self.Wu is not None, f"Wu has not been initialised, please call .fit"

        u_size = self.Wu.shape[1] # Input vector size
        x = np.zeros((x_size, 1))
        states = []
        for step in range(n_steps):
            # print(input.shape, u_size, self.Wu.shape)
            step_data = input[:, step].reshape((u_size, 1))

            # print(self.Wu.shape)
            # print(step_data)
            # print(np.vstack((1, step_data)))

            u = np.dot(self.Wu, step_data)
            # u = np.dot(self.Wu, np.vstack((1, step_data)))
            u[remove_node_idxs] = 0.

            x = np.tanh(u + np.dot(self.Wx, x))
            states += [x.flatten()]

            # if step == 0:
            #     x_state = (x)
            # else:
            #     x_state = np.hstack((x_state, x))
        return states


if __name__ == "__main__":
    fly_mat = pd.read_csv('/Users/alexdavies/Projects/whatIsLarva/science.add9330_data_s1_to_s4/Supplementary-Data-S1/all-all_connectivity_matrix.csv').drop(columns=['Unnamed: 0'])
    fly_mat = fly_mat.to_numpy()

    fly_mat[fly_mat > 0] = 1
    fly_mat[fly_mat != 1] = 0
    graph = fly_mat
    print(graph.shape, np.sum(graph))

    # graph = nx.fast_gnp_random_graph(10000, 0.025)
    # graph = nx.to_numpy_array(graph)

    steps = 1000

    train_ratio = 0.75
    n_in_train = int(train_ratio*steps)

    ts = np.linspace(0,4*np.pi, num=steps)


    frequencies = np.cos(ts)
    amp_saved = np.sin(2*np.pi*frequencies)

    amplitudes = np.copy(amp_saved).reshape((1, -1))
    amplitudes = np.concatenate((amplitudes, amplitudes), axis=0)
    print(frequencies.shape)

    frequencies_test = frequencies
    frequencies = frequencies[:n_in_train]

    amplitudes_test = amplitudes
    amplitudes      = amplitudes[:,:n_in_train]

    amp_saved_test = amp_saved
    amp_saved      = amp_saved[:n_in_train]


    model = RandomForestRegressor
    model_kwargs = {"n_estimators":500}

    # model = MLPRegressor
    # model_kwargs = {"hidden_layer_sizes":(200)}

    # model = SVR
    # model_kwargs = {"kernel":"linear"}

    res = Reservoir(graph, prediction_model=model, prediction_model_kwargs=model_kwargs)
    res.fit(amplitudes, frequencies)
    prediction = res.predict(amplitudes_test)
    # prediction_train = res.predict(amplitudes)

    default_regressor = model(**model_kwargs).fit(amplitudes.transpose(), frequencies)
    default_predicted = default_regressor.predict(amplitudes_test.transpose())
    # default_predicted_train = default_regressor.predict(amplitudes.transpose())

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))


    ax1.plot(ts, amp_saved_test)

    # ax2.plot(ts, frequencies_test, label = "Real")
    # ax2.plot(ts[200:], prediction, label = "Predicted")
    # ax2.plot(ts[200:], default_predicted, label="Predicted (Default)")


    ax2.plot(ts, frequencies_test, label = "Real", c = "black")
    ax2.scatter(ts, prediction, label = "Predicted", s = 2, alpha = 0.5)
    ax2.scatter(ts, default_predicted, label="Predicted (Default)", s = 2, alpha = 0.5)
    ax2.legend()
    plt.show()


