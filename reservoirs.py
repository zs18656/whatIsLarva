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

class ReservoirTorch:
    def __init__(self,
                 graph,
                 activation = torch.tanh,
                 spectral_radius = 0.9,
                 sparsity = None,
                 pca_reduce = False,
                 prediction_model = MLPRegressor,
                 prediction_model_kwargs = {"verbose":True, "hidden_layer_sizes":(20,)},
                 input_mask = None,
                 output_mask = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adjacency = torch.Tensor(graph).to(self.device)
        self.prediction_model = prediction_model(**prediction_model_kwargs)
        # Wu should be initialised when .fit is called
        self.Wx = self.create_weights(spectral_radius=spectral_radius,
                                      sparsity=sparsity).to(self.device)
        self.activation = activation

        self.input_mask = torch.ones(self.adjacency.shape[0], dtype=bool) if input_mask is None else input_mask
        self.output_mask = torch.ones(self.adjacency.shape[0], dtype=bool) if output_mask is None else output_mask
        self.input_mask, self.output_mask = self.input_mask.to(self.device), self.output_mask.to(self.device)

        self.Wu = None
        self.pca_reduce = pca_reduce
        self.pos = None

        if self.pca_reduce:
            self.pca = PCA(n_components=10)





    def create_weights(self, low=-1.0, high=1.0, sparsity=None, spectral_radius=None):
        shape = tuple(self.adjacency.shape)
        w = (high - low) * torch.rand(shape[0] * shape[1]).reshape(shape) + low  # create the weight matrix
        w[self.adjacency == 0] = 0.
        if not sparsity is None:  # if sparsity is defined
            s = torch.rand(shape[0] * shape[1]).reshape(shape) < (1.0 - sparsity)  # create a sparse boolean matrix
            w *= s  # set weight matrix values to 0.0
        if not spectral_radius is None:  # if spectral radius is defined
            sp = torch.max(torch.abs(torch.linalg.eig(w)[0]))  # compute current spectral radius
            w *= (spectral_radius) / sp  # adjust weight matrix to acheive specified spectral radius
        return w


    def vis_graph(self, weights = None, name = "Reservoir_Graph", node_colours = None):
        if weights is not None:
            G = nx.from_numpy_matrix(np.matrix(weights), create_using=nx.Graph)
        else:
            G = nx.from_numpy_matrix(np.matrix(self.adjacency), create_using=nx.Graph)

        node_colours = node_colours if node_colours is not None else np.ones(G.order())


        if self.pos is None:
            print(f"Calculating positions for {G}")
            self.pos = nx.nx_pydot.graphviz_layout(G, prog = "sfdp")

        fig, ax = plt.subplots(figsize=(9,9))
        nx.draw_networkx_nodes(G, pos = self.pos,
                               node_size = 4*np.abs(node_colours),
                               node_color=node_colours,
                               cmap = "bwr")
        if weights is not None:
            weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
            print(f"Drawing edges")
            nx.draw_networkx_edges(G, pos = self.pos, width=weights, ax = ax)
        else:
            nx.draw_networkx_edges(G, pos = self.pos,  width = 0.1, ax = ax)

        plt.savefig(f"{name}.png", dpi=300)
        plt.close()
        # plt.show()

    def fit(self, series, targets):
        self.Wu = 0.25*torch.ones([self.Wx.shape[0], series.shape[0]]).to(self.device)

        # Essentially training data, currently no warm-up
        embedded_states = self.forward(torch.Tensor(series).to(self.device))
        #
        # if self.pca_reduce:
        #     pca_embedded = self.pca.fit_transform(embedded_states)
        # else:
        #     pca_embedded = embedded_states
        # print(embedded_states)
        self.prediction_model.fit(torch.transpose(embedded_states, 0, 1).to("cpu"), targets)
        # plot_tree(self.prediction_model)
        # plt.show()


    def predict(self, series):
        assert self.Wu is not None, "Input weight matrix doesn't exist, must call .fit before .predict"
        embedded_states = self.forward(torch.Tensor(series).to(self.device))

        # if self.pca_reduce:
        #     pca_embedded = self.pca.transform(embedded_states)
        # else:
        #     pca_embedded = embedded_states

        prediction = self.prediction_model.predict(torch.transpose(embedded_states, 0, 1).to("cpu"))

        return prediction, embedded_states


    # Alex's function for a simple reservoir operation - please sense check!
    def forward(self, input, all_states = False):
        # Input should be shape |V| x n_steps
        n_steps = input.shape[1]
        x_size = self.Wx.shape[0] # Network size

        assert self.Wu is not None, f"Wu has not been initialised, please call .fit"
        output_dim = torch.sum(self.output_mask) if not all_states else n_steps
        x = torch.zeros((x_size, 1))
        states = torch.zeros((output_dim, n_steps))
        if all_states:
            out_mask = torch.ones(output_dim)
        else:
            out_mask = self.output_mask

        for step in tqdm(range(n_steps), leave=False):
            step_data = input[:, step].reshape((self.Wu.shape[1], 1))
            u = self.Wu @ step_data
            u[~self.input_mask] = 0.
            x = self.activation(u + (self.Wx + torch.eye(x_size)) @ x)
            states[:, step] = x.flatten()[out_mask]


        return states

class Reservoir:
    def __init__(self,
                 graph,
                 spectral_radius = 0.9,
                 sparsity = None,
                 pca_reduce = False,
                 prediction_model = MLPRegressor,
                 prediction_model_kwargs = {"verbose":True, "hidden_layer_sizes":(20,)},
                 input_mask = None,
                 output_mask = None):

        self.adjacency = graph
        self.prediction_model = prediction_model(**prediction_model_kwargs)
        # Wu should be initialised when .fit is called
        self.Wx = self.create_weights(spectral_radius=spectral_radius,
                                      sparsity=sparsity)

        self.input_mask = np.ones(self.adjacency.shape[0], dtype=bool) if input_mask is None else input_mask
        self.output_mask = np.ones(self.adjacency.shape[0], dtype=bool) if output_mask is None else output_mask

        self.Wu = None
        self.pca_reduce = pca_reduce
        self.pos = None

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


        if self.pos is None:
            print(f"Calculating positions for {G}")
            self.pos = nx.nx_pydot.graphviz_layout(G, prog = "sfdp")

        fig, ax = plt.subplots(figsize=(9,9))
        nx.draw_networkx_nodes(G, pos = self.pos,
                               node_size = 4*np.abs(node_colours),
                               node_color=node_colours,
                               cmap = "bwr")
        if weights is not None:
            weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
            print(f"Drawing edges")
            nx.draw_networkx_edges(G, pos = self.pos, width=weights, ax = ax)
        else:
            nx.draw_networkx_edges(G, pos = self.pos,  width = 0.1, ax = ax)

        plt.savefig(f"{name}.png", dpi=300)
        plt.close()
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
    def forward(self, input, all_states = False):
        # TODO: numpy-ise this (x, states), or even more to GPU with torch?
        # Input should be shape |V| x n_steps
        n_steps = input.shape[1]
        x_size = self.Wx.shape[0] # Network size

        assert self.Wu is not None, f"Wu has not been initialised, please call .fit"

        x = np.zeros((x_size, 1))
        states = []
        for step in range(n_steps):
            step_data = input[:, step].reshape((self.Wu.shape[1], 1))
            u = np.dot(self.Wu, step_data)
            # if input_mask is not None:
            u[~self.input_mask] = 0.
            x = np.tanh(u + np.dot((self.Wx+np.identity(x_size)), x))
            # if output_mask is not None:
            if not all_states:
                states += [x.flatten()[self.output_mask]]
            else:
                states += [x.flatten()]

        return states


class Optimiser:
    def __init__(self, res,
                         n_individuals = 240,
                         n_epochs = 1000,
                         keep_top = 3,
                         mutation_noise = 0.01,
                         input_size = 100,
                         output_size = 100,
                         n_workers = 12):
        self.res = res
        self.dimensions = self.res.adjacency.shape[0]
        self.n_individuals = n_individuals
        self.n_epochs = n_epochs
        self.keep_top = keep_top
        self.graph = nx.from_numpy_matrix(self.res.adjacency)
        self.mutation_noise = mutation_noise
        self.input_size = input_size
        self.output_size = output_size
        self.n_workers = n_workers
        pass
    
    def random_mask(self, mode = "output"):
        mask  = np.zeros(self.dimensions, dtype = bool)

        if mode == "output":
            mask_dim = self.output_size
        else:
            mask_dim = self.input_size
        mask[np.random.randint(low = 0, high = self.dimensions, size=mask_dim)] = True
        
        return mask
    def optimise_inputs(self,
                        series_train, targets_train,
                        series_test, targets_test):

        masks = [self.random_mask() for _ in range(self.n_individuals)]

        n_repeats = self.n_individuals // self.keep_top
        remainder = self.n_individuals % self.keep_top

        pbar = tqdm(range(self.n_epochs), leave = False)

        for i_epoch in pbar:
            masks, min_mse = self.measure_performance_inputs_per_mask(masks,
                                                                      series_train,
                                                                      targets_train,
                                                                      series_test,
                                                                      targets_test)
            pbar.set_description(f"Epoch {i_epoch}, Minimum MSE: {min_mse}")
            top_n_masks = masks[:self.keep_top]
            masks = []

            for i_mask, mask in enumerate(top_n_masks):
                if i_mask == 0:
                    masks += [mask] * (n_repeats+remainder)
                else:
                    masks += [mask] * n_repeats

            if i_epoch == self.n_epochs:
                break

            masks = [self.mutate(mask) for mask in masks]

        print(f"Final Minimum MSE: {min_mse}")
        self.res.input_mask = masks[0]
        self.res.fit(series_train, targets_train)
        return self.res

    def optimise_outputs(self,
                        series_train, targets_train,
                        series_test, targets_test):

        masks = [self.random_mask(mode="output") for _ in range(self.n_individuals)]
        # Have to call .fit before optimising
        self.res.fit(series_train, targets_train)

        n_repeats = self.n_individuals // self.keep_top
        remainder = self.n_individuals % self.keep_top

        pbar = tqdm(range(self.n_epochs), leave = False)

        states_train = self.res.forward(series_train, all_states = True)
        states_test = self.res.forward(series_test, all_states=True)

        mses  = []

        for i_epoch in pbar:
            masks, min_mse = self.measure_performance_outputs_per_mask(masks, states_train, targets_train, states_test, targets_test)
            mses.append(min_mse)
            pbar.set_description(f"Epoch {i_epoch}, Minimum MSE: {min_mse}")
            top_n_masks = masks[:self.keep_top]
            masks = []

            for i_mask, mask in enumerate(top_n_masks):
                if i_mask == 0:
                    masks += [mask] * (n_repeats+remainder)
                else:
                    masks += [mask] * n_repeats

            if i_epoch == self.n_epochs:
                break

            masks = [self.mutate(mask) for mask in masks]

        # plt.plot(mses)
        # plt.show()
        # quit()
        print(mses)

        print(f"Final Minimum MSE: {min_mse}")
        self.res.output_mask = masks[0]
        self.res.fit(series_train, targets_train)
        return self.res


    def per_input_mask_mse(self,
                           packet
                           ):

        mask, series_train, targets_train, series_test, targets_test, ind = packet

        self.res.input_mask = mask
        self.res.fit(series_train, targets_train)

        mse = np.mean((self.res.predict(series_test)[0] - targets_test) ** 2)

        return mse, ind

    def measure_performance_inputs_per_mask(self, masks,
                        series_train, targets_train,
                        series_test, targets_test):

        mses = np.zeros(len(masks))

        packets = [[mask, series_train, targets_train, series_test, targets_test, i] for i, mask in enumerate(masks)]
        # pbar = tqdm(masks, leave = False)

        with cf.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []

            for i, packet in enumerate(packets):
                futures.append(executor.submit(self.per_input_mask_mse, packet = packet))

            for future in cf.as_completed(futures):
                mse, ind = future.result()
                mses[ind] = mse

        # for i_mask, input_mask in enumerate(masks):
        #
        #     self.res.input_mask = input_mask
        #     self.res.fit(series_train, targets_train)
        #
        #     mse = np.sum((self.res.predict(series_test)[0] - targets_test) ** 2)
        #     # pbar.set_description(f"MSE: {mse}")
        #
        #     mses[i_mask] = mse

        sorting_inds = np.argsort(mses)

        return [masks[ind] for ind in sorting_inds], np.min(mses)
        # self.res.input_mask = best_input
        # self.res.fit(series_train, targets_train)
        # return self.res

    def measure_performance_outputs_per_mask(self, masks,
                        states_train, targets_train,
                        states_test, targets_test):

        mses = np.zeros(len(masks))
        # pbar = tqdm(masks, leave = False)

        for i_mask, mask in enumerate(masks):
            these_states_train = [state[mask] for state in states_train]
            these_states_test = [state[mask] for state in states_test]

            # self.res.output_mask = output_mask
            # self.res.fit(series_train, targets_train)

            self.res.prediction_model.fit(these_states_train, targets_train)

            mse = np.mean((self.res.prediction_model.predict(these_states_test)[0] - targets_test) ** 2)
            mses[i_mask] = mse
            # pbar.set_description(f"MSE: {mse}, best {best_mse}")

            # if mse < best_mse:
            #     best_output = output_mask
            #     best_mse = mse

        sorting_inds = np.argsort(mses)

        return [masks[ind] for ind in sorting_inds], np.min(mses)
        # self.res.input_mask = best_input
        # self.res.fit(series_train, targets_train)
        # return self.res



    # def optimise_inputs(self, series_train, targets_train,
    #              series_test, targets_test,
    #              iterations = 10):
    #
    #     dimensions = self.res.adjacency.shape[0]
    #
    #     best_mse = 1e3
    #     best_input = None
    #     pbar = tqdm(range(iterations), leave = False)
    #     for i in pbar:
    #         input_mask  = np.zeros(dimensions, dtype = bool)
    #
    #         input_mask[np.random.randint(low = 0, high = dimensions, size=self.input_size)] = True
    #
    #         self.res.input_mask = input_mask
    #         self.res.fit(series_train, targets_train)
    #
    #         mse = np.sum((self.res.predict(series_test)[0] - targets_test) ** 2)
    #         pbar.set_description(f"MSE: {mse}, best {best_mse}")
    #
    #         if mse < best_mse:
    #             best_input = input_mask
    #             best_mse = mse
    #
    #     self.res.input_mask = best_input
    #     self.res.fit(series_train, targets_train)
    #     return self.res
    #
    # def optimise_outputs(self, series_train, targets_train,
    #              series_test, targets_test,
    #              iterations = 100):
    #
    #     dimensions = self.res.adjacency.shape[0]
    #
    #     best_mse = 1e3
    #     best_output = None
    #     pbar = tqdm(range(iterations), leave = False)
    #
    #     states_train = self.res.forward(series_train, all_states = True)
    #     states_test = self.res.forward(series_test, all_states=True)
    #     for i in pbar:
    #         output_mask  = np.zeros(dimensions, dtype = bool)
    #         output_mask[np.random.randint(low = 0, high = dimensions, size=self.output_size)] = True
    #
    #         these_states_train = [state[output_mask] for state in states_train]
    #         these_states_test = [state[output_mask] for state in states_test]
    #
    #         # self.res.output_mask = output_mask
    #         # self.res.fit(series_train, targets_train)
    #
    #         self.res.prediction_model.fit(these_states_train, targets_train)
    #
    #         mse = np.sum((self.res.prediction_model.predict(these_states_test)[0] - targets_test) ** 2)
    #         pbar.set_description(f"MSE: {mse}, best {best_mse}")
    #
    #         if mse < best_mse:
    #             best_output = output_mask
    #             best_mse = mse
    #
    #     self.res.output_mask = best_output
    #     self.res.fit(series_train, targets_train)
    #     return self.res

    def mutate(self, mask):
        # Mask: Boolean numpy array for whether a node is active

        node_ids = np.argwhere(mask != 0)
        random_selection = np.random.random(node_ids.size)
        selected_nodes = node_ids[np.argwhere(random_selection <= self.mutation_noise)]
        mask[selected_nodes] = 0.

        for node in selected_nodes:
            # print(self.graph, node[0][0])
            neighbours = np.array(list(self.graph.neighbors(node[0][0])))
            # print(neighbours)
            new_node_id = neighbours[np.random.randint(low = 0, high = len(neighbours), size=1)]

            mask[new_node_id] = 1.

        return mask





def vis_with_states(res, series, targets, directory = "frames"):
    if directory not in os.listdir():
        os.mkdir(directory)
    os.chdir(directory)

    states = res.forward(series, all_states = True)
    pbar = tqdm(states)
    for i, state in enumerate(pbar):
        pbar.set_description(f"Min: {np.min(state)}, Max: {np.max(state)}")
        res.vis_graph(node_colours=state, name=f"{i}")

        # fig, ax = plt.subplots(figsize=(4,3))
        # print(state)
        # ax.hist(state, bins = 100)
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([0,300])
        # plt.savefig(f"{i}.png")

        # plt.close()

    # Build GIF
    with imageio.get_writer('gif.gif', mode='I') as writer:
        for filename in os.listdir():
            if "png" in filename:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)

    os.chdir('../')







if __name__ == "__main__":
    fly_mat = pd.read_csv('/Users/alexdavies/Projects/whatIsLarva/science.add9330_data_s1_to_s4/Supplementary-Data-S1/all-all_connectivity_matrix.csv').drop(columns=['Unnamed: 0'])
    fly_mat = fly_mat.to_numpy()
    #

    # Could be fun to trim only to multiple-synapse connections?
    fly_mat[fly_mat > 1] = 1
    fly_mat[fly_mat != 1] = 0
    fly_mat[np.identity(fly_mat.shape[0], dtype=bool)] = 0.
    fly_graph = fly_mat
    print(fly_graph.shape, np.sum(fly_graph))

    rand_graph = nx.fast_gnp_random_graph(fly_graph.shape[0], np.sum(fly_graph) / (fly_graph.shape[0] ** 2))
    rand_graph = nx.to_numpy_array(rand_graph)

    fly_nx = nx.from_numpy_matrix(fly_graph, create_using=nx.Graph)
    print(fly_nx)
    bter_start = time()
    bt = BTER(fly_nx)
    bt.fit()
    bter_graph = bt.sample()
    print(f"BTER sampling and fitting took {time() - bter_start}s")

    Gcc = sorted(nx.connected_components(bter_graph), key=len, reverse=True)
    bter_graph = bter_graph.subgraph(Gcc[0])

    bter_graph = nx.to_numpy_array(bter_graph)

    bter_graph[np.identity(bter_graph.shape[0], dtype=bool)] = 0.

    print(f"Fly graph {np.sum(fly_graph)} edges, bter graph {np.sum(bter_graph)} edges, rand graph {np.sum(rand_graph)} edges")

    steps = 3200

    train_ratio = 0.75
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

    # model = DecisionTreeRegressor
    # model_kwargs = {"max_depth":5}

    # model = RandomForestRegressor
    # model_kwargs = {"n_estimators":50, "n_jobs":6, "max_depth":6}

    model = MLPRegressor
    model_kwargs = {"hidden_layer_sizes":(20),
                    "activation":"tanh",
                    }

    # model = LinearRegression
    # model_kwargs = {"n_jobs":4}

    # model = SVR
    # model_kwargs = {"kernel":"linear"}

    model_name = str(model).split('.')[-1].split("'")[0]

    # opt = Optimiser(fly_res)
    # fly_res = opt.optimise_inputs(amplitudes, frequencies,
    #              amplitudes_test, frequencies_test)
    # fly_res = opt.optimise_outputs(amplitudes, frequencies,
    #              amplitudes_test, frequencies_test)




    # opt = Optimiser(rand_res)
    # rand_res = opt.optimise_inputs(amplitudes, frequencies,
    #              amplitudes_test, frequencies_test)
    # rand_res = opt.optimise_outputs(amplitudes, frequencies,
    #              amplitudes_test, frequencies_test)


    torch_start = time()
    fly_res = ReservoirTorch(fly_graph,
                             prediction_model=model,
                             prediction_model_kwargs=model_kwargs,
                             activation=torch.sigmoid)
    rand_res = ReservoirTorch(rand_graph,
                              prediction_model=model,
                              prediction_model_kwargs=model_kwargs,
                              activation=torch.sigmoid)
    bter_res = ReservoirTorch(bter_graph,
                              prediction_model=model,
                              prediction_model_kwargs=model_kwargs,
                              activation=torch.sigmoid)

    fly_res.fit(amplitudes, frequencies)
    rand_res.fit(amplitudes, frequencies)
    bter_res.fit(amplitudes, frequencies)

    print(f"\n\ntorch fitting time {time() - torch_start}")

    # numpy_start = time()
    # fly_res = Reservoir(fly_graph, prediction_model=model, prediction_model_kwargs=model_kwargs)
    # rand_res = Reservoir(rand_graph, prediction_model=model, prediction_model_kwargs=model_kwargs)
    #
    # fly_res.fit(amplitudes, frequencies)
    # rand_res.fit(amplitudes, frequencies)
    #
    # print(f"numpy fitting time {time() - numpy_start}\n\n")

    # vis_with_states(fly_res, amplitudes_test, frequencies_test, directory="frames_graph")

    # plt.hist([fly_res.prediction_model.feature_importances_,rand_res.prediction_model.feature_importances_],
    #          label = ["fly","random"],
    #          bins=100)
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # # plt.hist(, label = "random")
    # plt.show()
    # quit()

    # print(fly_res.prediction_model.feature_importances_)
    # print(rand_res.prediction_model.feature_importances_)

    # fly_res.vis_graph(weights=None, name="fly", node_colours=fly_res.prediction_model.feature_importances_)
    # rand_res.vis_graph(weights=None, name="random", node_colours=rand_res.prediction_model.feature_importances_)

    # fly_res.vis_graph(weights=None, name="fly", node_colours=-0.5 * fly_res.input_mask + 0.5 * fly_res.output_mask)
    # rand_res.vis_graph(weights=None, name="random", node_colours=-0.5*rand_res.input_mask + 0.5*rand_res.output_mask)

    fly_prediction, fly_states = fly_res.predict(amplitudes_test)
    rand_prediction, rand_states = rand_res.predict(amplitudes_test)
    bter_prediction, bter_states = bter_res.predict(amplitudes_test)
    # prediction_train = res.predict(amplitudes)

    # default_regressor = model(**model_kwargs).fit(amplitudes.transpose(), frequencies)
    # default_predicted = default_regressor.predict(amplitudes_test.transpose())
    # default_predicted_train = default_regressor.predict(amplitudes.transpose())

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(12,12))

    ax1.set_title("Input signal")
    ax2.set_title("Std. Dev. of Reservoir State")
    ax3.set_title("Frequency Prediction")
    ax4.set_title(f"Squared Error")

    ax1.plot(ts, amp_saved_test)
    ax1.axvline(ts[n_in_train], c = "black", linestyle = ":")
    ax2.axvline(ts[n_in_train], c = "black", linestyle = ":")
    ax2.scatter(ts, torch.std(fly_states, dim = 0), s = 2, c = "green", label = "Fly States")
    ax2.scatter(ts, torch.std(rand_states, dim = 0), s = 2, c="blue", label="Random States")
    ax2.scatter(ts, torch.std(bter_states, dim=0), s=2, c="red", label="BTER States")

    # ax2.plot(ts, frequencies_test, label = "Real")
    # ax2.plot(ts[200:], prediction, label = "Predicted")
    # ax2.plot(ts[200:], default_predicted, label="Predicted (Default)")
    ax3.axvline(ts[n_in_train], c = "black", linestyle = ":")
    ax3.plot(ts, frequencies_test, label = "Real", c = "black")
    ax3.scatter(ts, fly_prediction, label = "Fly Predicted", s = 4, alpha = 0.75, c = "green")
    ax3.scatter(ts, rand_prediction, label="Random Predicted", s=2, alpha=0.75, c = "blue")
    ax3.scatter(ts, bter_prediction, label="Random Predicted", s=2, alpha=0.75, c="red")
    # ax3.scatter(ts, default_predicted, label=f"Predicted {model_name}", s = 2, alpha = 0.5, color = "red")
    # ax3.legend()

    # ax4.plot(ts, frequencies_test, label = "Real", c = "black")
    ax4.axvline(ts[n_in_train], c = "black", linestyle = ":", label = "Train Cutoff")

    ax4.axhline(np.mean((fly_prediction - frequencies_test)**2), c="green", linestyle=":", label=f"MSE Fly {str(np.mean((fly_prediction - frequencies_test)**2))[:5]}")
    ax4.axhline(np.mean((rand_prediction - frequencies_test)**2), c="blue", linestyle=":", label=f"MSE Rand {str(np.mean((rand_prediction - frequencies_test)**2))[:5]}")
    ax4.axhline(np.mean((bter_prediction - frequencies_test) ** 2), c="red", linestyle=":", label=f"MSE BTER {str(np.mean((rand_prediction - frequencies_test) ** 2))[:5]}")

    ax4.scatter(ts, (fly_prediction - frequencies_test)**2, label = "Fly Predicted", s = 10, alpha = 0.75, c = "green")
    ax4.scatter(ts, (rand_prediction - frequencies_test)**2, label="Random Predicted", s=5, alpha=0.75, c = "blue")
    ax4.scatter(ts, (bter_prediction - frequencies_test) ** 2, label="Random Predicted", s=5, alpha=0.75, c="red")
    ax4.set_yscale('log')
    # ax4.plot(ts, (default_predicted - frequencies_test)**2, label=f"Predicted {model_name}",  alpha = 0.5, color = "red")
    ax4.legend()


    plt.savefig(f"{datetime.now()}.png")
    plt.show()


