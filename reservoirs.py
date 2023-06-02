import numpy as np
import pandas as pd # only used to read the MNIST data set
import networkx as nx
from datetime import datetime
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
from sklearn.decomposition import PCA

from sklearn.inspection import permutation_importance

from sklearn.svm import SVC, LinearSVC

import os
os.path.abspath(os.getcwd())

class Reservoir:
    def __init__(self,
                 graph,
                 spectral_radius = 0.9,
                 sparsity = None,
                 pca_reduce = False,
                 prediction_model = MLPRegressor,
                 prediction_model_kwargs = {"verbose":True, "hidden_layer_sizes":(20,)}):

        self.adjacency = graph
        self.prediction_model = prediction_model(**prediction_model_kwargs)
        # Wu should be initialised when .fit is called
        self.Wx = self.create_weights(spectral_radius=spectral_radius,
                                      sparsity=sparsity)
        self.Wu = None
        self.pca_reduce = pca_reduce

        if self.pca_reduce:
            self.pca = PCA(n_components=10)



    def create_weights(self, low=-1.0, high=1.0, sparsity=None, spectral_radius=None):
        shape = tuple(self.adjacency.shape)
        w = (high - low) * np.random.ranf(shape[0] * shape[1]).reshape(shape) + low  # create the weight matrix
        w[self.adjacency == 0] = 0.
        if not sparsity is None:  # if sparsity is defined
            s = np.random.ranf(shape[0] * shape[1]).reshape(shape) < (1.0 - sparsity)  # create a sparse boolean matrix
            w *= s  # set weight matrix values to 0.0
        if not spectral_radius is None:  # if spectral radius is defined
            sp = np.max(np.abs(np.linalg.eig(w)[0]))  # compute current spectral radius
            w *= (spectral_radius) / sp  # adjust weight matrix to acheive specified spectral radius
        return w


    def vis_graph(self, weights = None, name = "Reservoir_Graph", node_colours = None):
        if weights is not None:
            G = nx.from_numpy_matrix(np.matrix(weights), create_using=nx.Graph)
        else:
            G = nx.from_numpy_matrix(np.matrix(self.adjacency), create_using=nx.Graph)

        node_colours = node_colours if node_colours is not None else np.ones(G.order())

        print(f"Calculating positions for {G}")
        pos = nx.nx_pydot.graphviz_layout(G, prog = "sfdp")

        fig, ax = plt.subplots(figsize=(9,9))
        nx.draw_networkx_nodes(G, pos = pos,
                               node_size=100*node_colours,
                               node_color=node_colours)
        if weights is not None:
            weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
            print(f"Drawing edges")
            nx.draw_networkx_edges(G, pos = pos, width=weights, alpha = np.abs(weights), ax = ax)
        else:
            nx.draw_networkx_edges(G, pos = pos, alpha=0.1, width = 0.1, ax = ax)

        plt.savefig(f"{name}.png", dpi=600)
        # plt.show()

    def fit(self, series, targets):
        self.Wu = 0.25*np.ones([self.Wx.shape[0], series.shape[0]])

        # Essentially training data, currently no warm-up
        embedded_states = self.forward(series)
        if self.pca_reduce:
            pca_embedded = self.pca.fit_transform(embedded_states)
        else:
            pca_embedded = embedded_states
        # print(embedded_states)

        self.prediction_model.fit(pca_embedded, targets)
        # plot_tree(self.prediction_model)
        # plt.show()


    def predict(self, series):
        assert self.Wu is not None, "Input weight matrix doesn't exist, must call .fit before .predict"
        embedded_states = self.forward(series)
        if self.pca_reduce:
            pca_embedded = self.pca.transform(embedded_states)
        else:
            pca_embedded = embedded_states
        prediction = self.prediction_model.predict(pca_embedded)

        return prediction, embedded_states


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

            x = np.tanh(u + np.dot((self.Wx+np.identity(x_size)), x))
            states += [x.flatten()]

            # if step == 0:
            #     x_state = (x)
            # else:
            #     x_state = np.hstack((x_state, x))
        return states


if __name__ == "__main__":
    fly_mat = pd.read_csv('/Users/alexdavies/Projects/whatIsLarva/science.add9330_data_s1_to_s4/Supplementary-Data-S1/all-all_connectivity_matrix.csv').drop(columns=['Unnamed: 0'])
    fly_mat = fly_mat.to_numpy()
    #
    fly_mat[fly_mat > 0] = 1
    fly_mat[fly_mat != 1] = 0
    fly_graph = fly_mat
    # print(graph.shape, np.sum(graph))

    rand_graph = nx.fast_gnp_random_graph(fly_graph.shape[0], np.sum(fly_graph) / (fly_graph.shape[0] ** 2))
    rand_graph = nx.to_numpy_array(rand_graph)

    print(f"Fly graph {np.sum(fly_graph)} edges, rand graph {np.sum(rand_graph)} edges")

    steps = 200

    train_ratio = 0.5
    n_in_train = int(train_ratio*steps)

    ts = np.linspace(0,8*np.pi, num=steps)

    t_noise = np.random.randn(steps)
    # Add more noise in non-train section
    t_noise *= 0.025
    # t_noise[n_in_train:] *= 0.05


    frequencies = np.cos(ts)
    amp_saved = np.sin(np.pi*frequencies)  + t_noise

    amplitudes = np.copy(amp_saved).reshape((1, -1))
    amplitudes = np.concatenate((amplitudes, amplitudes), axis=0)
    # print(frequencies.shape)

    frequencies_test = frequencies
    frequencies = frequencies[:n_in_train]

    amplitudes_test = amplitudes
    amplitudes      = amplitudes[:,:n_in_train]

    amp_saved_test = amp_saved
    amp_saved      = amp_saved[:n_in_train]


    model = RandomForestRegressor
    model_kwargs = {"n_estimators":100, "verbose":1, "n_jobs":6}

    # model = MLPRegressor
    # model_kwargs = {"hidden_layer_sizes":(100)}

    # model = SVR
    # model_kwargs = {"kernel":"linear"}

    model_name = str(model).split('.')[-1].split("'")[0]

    fly_res = Reservoir(fly_graph, prediction_model=model, prediction_model_kwargs=model_kwargs)
    rand_res = Reservoir(rand_graph, prediction_model=model, prediction_model_kwargs=model_kwargs)


    # quit()

    fly_res.fit(amplitudes, frequencies)
    rand_res.fit(amplitudes, frequencies)

    # plt.hist([fly_res.prediction_model.feature_importances_,rand_res.prediction_model.feature_importances_],
    #          label = ["fly","random"],
    #          bins=100)
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # # plt.hist(, label = "random")
    # plt.show()
    # quit()

    print(fly_res.prediction_model.feature_importances_)
    print(rand_res.prediction_model.feature_importances_)

    fly_res.vis_graph(weights=None, name="fly", node_colours=fly_res.prediction_model.feature_importances_)
    rand_res.vis_graph(weights=None, name="random", node_colours=rand_res.prediction_model.feature_importances_)

    fly_prediction, fly_states = fly_res.predict(amplitudes_test)
    rand_prediction, rand_states = rand_res.predict(amplitudes_test)
    # prediction_train = res.predict(amplitudes)

    default_regressor = model(**model_kwargs).fit(amplitudes.transpose(), frequencies)
    default_predicted = default_regressor.predict(amplitudes_test.transpose())
    # default_predicted_train = default_regressor.predict(amplitudes.transpose())

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(12,12))

    ax1.set_title("Input signal")
    ax2.set_title("Std. Dev. of Reservoir State")
    ax3.set_title("Frequency Prediction")
    ax4.set_title(f"MSE")

    ax1.plot(ts, amp_saved_test)

    ax2.scatter(ts, [np.std(state) for state in fly_states], s = 2, c = "green", label = "Fly States")
    ax2.scatter(ts, [np.std(state) for state in rand_states], s = 2, c="blue", label="Random States")

    # ax2.plot(ts, frequencies_test, label = "Real")
    # ax2.plot(ts[200:], prediction, label = "Predicted")
    # ax2.plot(ts[200:], default_predicted, label="Predicted (Default)")

    ax3.plot(ts, frequencies_test, label = "Real", c = "black")
    ax3.scatter(ts, fly_prediction, label = "Fly Predicted", s = 4, alpha = 0.75, c = "green")
    ax3.scatter(ts, rand_prediction, label="Random Predicted", s=2, alpha=0.75, c = "blue")
    ax3.scatter(ts, default_predicted, label=f"Predicted {model_name}", s = 2, alpha = 0.5, color = "red")
    # ax3.legend()

    # ax4.plot(ts, frequencies_test, label = "Real", c = "black")
    ax4.scatter(ts, (fly_prediction - frequencies_test)**2, label = "Fly Predicted", s = 10, alpha = 0.75, c = "green")
    ax4.scatter(ts, (rand_prediction - frequencies_test)**2, label="Random Predicted", s=5, alpha=0.75, c = "blue")
    ax4.plot(ts, (default_predicted - frequencies_test)**2, label=f"Predicted {model_name}",  alpha = 0.5, color = "red")
    ax4.legend()


    plt.savefig(f"{datetime.now()}.png")
    plt.show()


