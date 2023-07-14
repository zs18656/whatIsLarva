import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import ESN
from tqdm import tqdm


# def create_weights(self, low=-1.0, high=1.0, sparsity=None, spectral_radius=None):
#     shape = tuple(self.adjacency.shape)
#     w = (high - low) * torch.rand(shape[0] * shape[1]).reshape(shape) + low  # create the weight matrix
#     w[self.adjacency == 0] = 0.
#
#     if not sparsity is None:  # if sparsity is defined
#         s = torch.rand(shape[0] * shape[1]).reshape(shape) < (1.0 - sparsity)  # create a sparse boolean matrix
#         w *= s  # set weight matrix values to 0.0
#     if not spectral_radius is None:  # if spectral radius is defined
#         sp = torch.max(torch.abs(torch.linalg.eig(w)[0]))  # compute current spectral radius
#         w *= (spectral_radius) / sp  # adjust weight matrix to acheive specified spectral radius
#     return w

def random_weights(graph, sr = None):
    # w = 2 * np.random.random(graph.shape) - 1
    # w = np.random.random(graph.shape)

    w = np.random.randn(*graph.shape)
    w[graph == 0] = 0.

    if not sr is None:  # if spectral radius is defined
        sp = np.max(np.abs(np.linalg.eig(w)[0]))  # compute current spectral radius
        w *= (sr) / sp  # adjust weight matrix to acheive specified spectral radius
    return w


    # return graph * random_assignment

def load_fly(return_tensor = False):
    pwd = os.getcwd()
    data_path = os.path.join(pwd, "science.add9330_data_s1_to_s4/Supplementary-Data-S1/all-all_connectivity_matrix.csv")
    # fly_mat = pd.read_csv('/Users/alexdavies/Projects/whatIsLarva/science.add9330_data_s1_to_s4/Supplementary-Data-S1/all-all_connectivity_matrix.csv').drop(columns=['Unnamed: 0'])
    fly_mat = pd.read_csv(
        data_path).drop(
        columns=['Unnamed: 0'])
    fly_mat = fly_mat.to_numpy()
    #

    # Could be fun to trim only to multiple-synapse connections?
    fly_mat[fly_mat > 1] = 1
    fly_mat[fly_mat != 1] = 0
    fly_mat[np.identity(fly_mat.shape[0], dtype=bool)] = 0.
    fly_graph = fly_mat
    print(f"Graph with {fly_graph.shape[0]} nodes and {np.sum(fly_graph)} edges")

    nx_graph = nx.from_numpy_array(fly_graph)

    degrees = nx_graph.degree()
    degrees = [d for _, d in degrees]
    plt.hist(degrees, bins = 100)
    plt.savefig("fly_degrees.png")
    plt.close()
    # plt.show()
    #
    # random_assignment = np.random.random(fly_graph.shape)
    # fly_graph *= random_assignment

    if return_tensor:
        return torch.Tensor(fly_graph)
    else:
        return fly_graph

def create_random(fly_graph, return_tensor = False):
    rand_graph = nx.fast_gnp_random_graph(fly_graph.shape[0], np.sum(fly_graph) / (fly_graph.shape[0] ** 2))
    print(rand_graph)

    degrees = rand_graph.degree()
    degrees = [d for _, d in degrees]
    plt.hist(degrees, bins = 100)
    plt.savefig("rand_degrees.png")
    plt.close()
    # plt.show()

    rand_graph = nx.to_numpy_array(rand_graph)

    print(f"Graph with {rand_graph.shape[0]} nodes and {np.sum(rand_graph)} edges")


    #
    # random_assignment = np.random.random(rand_graph.shape)
    # rand_graph *= random_assignment

    if return_tensor:
        return torch.Tensor(rand_graph)
    else:
        return rand_graph

def get_ESN(adjacency, cfg = None, use_defaults = True):

    if cfg is not None and use_defaults is False:

        sr = cfg["sr"]
        lr = cfg["lr"]
        iss = cfg["iss"]
        variable_seed = cfg["seed"]
        ridge = cfg["ridge"]

    # print(cfg)
        reservoir_graph = random_weights(adjacency, sr=sr)
        reservoir = Reservoir(W=reservoir_graph,
                              lr=lr,
                              input_scaling=iss,
                              seed=variable_seed)
        readout = Ridge(ridge=ridge)
    else:
        reservoir_graph = random_weights(adjacency, sr=1)
        reservoir = Reservoir(W=reservoir_graph)

        readout = Ridge(ridge = 1e-7)

    model = reservoir >> readout
    # model = ESN(reservoir=reservoir, readout=readout, workers=2)

    return model

def objective(dataset, config, *, iss, N, sr, lr, ridge, seed):

    # This step may vary depending on what you put inside 'dataset'
    train_data, validation_data = dataset
    X_train, y_train = train_data
    X_val, y_val = validation_data

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []; r2s = [];
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=variable_seed)

        readout = Ridge(ridge=ridge)

        model = reservoir >> readout


        # Train your model and test your model.
        predictions = model.fit(X_train, y_train) \
                           .run(X_test)

        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}

def objective_given_graph(dataset, config, *, graph_to_use, iss, sr, lr, ridge, seed):

    # This step may vary depending on what you put inside 'dataset'
    train_data, validation_data, graph = dataset
    X_train, y_train = train_data
    X_val, y_val = validation_data

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed
    # if graph_to_use == "fly":
    #     graph = load_fly()
    # else:
    #     graph = create_random(load_fly())
    n_neurons = graph.shape[0]
    # print(graph.shape)
        # graph = create_random(load_fly())

    losses = []; r2s = [];
    for n in range(instances):
        # Build your model given the input parameters

        # if graph_to_use == "fly":
        cfg = {'iss': iss, 'lr': lr,  'ridge': ridge, 'seed': seed, 'sr':sr}
        model = get_ESN(graph, cfg)

        # Train your model and test your model.
        predictions = model.fit(X_train, y_train) \
                           .run(X_test)

        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import doublescroll, lorenz, multiscroll, mackey_glass
from reservoirpy.observables import nrmse, rsquare

hyperopt_config_fly = {
    "exp": f"hyperopt-multiscroll-fly", # the experimentation name
    "hp_max_evals": 50,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "graph_to_use": ["choice", "fly"],
        "sr": ["loguniform", 1e-6, 10],   # the spectral radius is log-uniformly distributed between 1e-6 and 10
        "lr": ["loguniform", 1e-3, 1],    # idem with the leaking rate, from 1e-3 to 1
        "iss": ["uniform", 0.1, 0.9],           # the input scaling is fixed
        "ridge": ["loguniform", 1e-8, 1e-6],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}

hyperopt_config_rand = {
    "exp": f"hyperopt-multiscroll-rand", # the experimentation name
    "hp_max_evals": 50,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "graph_to_use": ["choice", "random"],
        "sr": ["loguniform", 1e-6, 10],   # the spectral radius is log-uniformly distributed between 1e-6 and 10
        "lr": ["loguniform", 1e-4, 1],    # idem with the leaking rate, from 1e-3 to 1
        "iss": ["uniform", 0.1, 0.9],           # the input scaling is fixed
        "ridge": ["loguniform", 1e-8, 1e-6],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}


import json
import reservoirpy as rpy
# rpy.verbosity(0)
from reservoirpy.hyper import research
from reservoirpy.hyper import plot_hyperopt_report

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config_fly['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config_fly, f)

with open(f"{hyperopt_config_rand['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config_rand, f)


timesteps = 800
warmup    = 100
# x0 = [0.37926545, 0.058339, -0.08167691]
# X = doublescroll(timesteps, x0=x0, method="RK23")
X = mackey_glass(timesteps)# lorenz(timesteps)

train_len = 600
X_train = X[:train_len]
y_train = X[1 : train_len + 1]
X_test = X[train_len : -1]
y_test = X[train_len + 1:]

fly_graph = load_fly()
fly_dataset = ((X_train, y_train), (X_test, y_test), fly_graph)
rand_graph = create_random(load_fly())
rand_dataset = ((X_train, y_train), (X_test, y_test), rand_graph)



# fly_best = {'input_scaling': 0.5872833192706555, 'lr': 0.15492403550483738, 'sr': 0.07071032186253305, 'iss': 0.8866121053359386}
# rand_best = {'input_scaling': 0.723792790570117, 'lr': 0.0236187254130727, 'sr': 0.0014791230744112718}

# Best params with normal weights
# fly_best = {'graph_to_use': 0, 'iss': 0.39829017376029685, 'lr': 0.32990593409366, 'rc_connectivity': 0, 'ridge': 0, 'seed': 0, 'sr': 0.0009233404270377002}
# rand_best = {'graph_to_use': 0, 'iss': 0.4828800948091627, 'lr': 0.31117781577679504, 'rc_connectivity': 0.05906443706327862, 'ridge': 0, 'seed': 0, 'sr': 0.0005419008715256282}


# fly_best = research(objective_given_graph, fly_dataset,  f"{hyperopt_config_fly['exp']}.config.json", ".")[0]
#
# rand_best = research(objective_given_graph, rand_dataset, f"{hyperopt_config_rand['exp']}.config.json", ".")[0]

# Quick testing for SR
fly_best = {'graph_to_use': 0, 'iss': 0.16652051240971302, 'lr': 0.9349616902731771, 'ridge': 2.1422683079778163e-08, 'seed': 0, 'sr': 0.36599676436640677}
rand_best = {'graph_to_use': 0, 'iss': 0.16652051240971302, 'lr': 0.9142362168242784, 'ridge': 2.1422683079778163e-08, 'seed': 0, 'sr': 0.36599676436640677}


# Random weights on fly
# fly_best = {'graph_to_use': 0, 'iss': 0.8866121053359386, 'lr': 0.15693214219369533, 'rc_connectivity': 0, 'ridge': 0, 'seed': 0, 'sr': 0.0011845471145886099}
# rand_best = {'graph_to_use': 0, 'iss': 0.24168746352116566, 'lr': 0.9487053215849356, 'rc_connectivity': 0.01646669730864767, 'ridge': 0, 'seed': 0, 'sr': 8.753193828153007}# research(objective_given_graph,  dataset, f"{hyperopt_config_rand['exp']}.config.json", ".")[0]
# fly_best = research(objective_given_graph,  dataset, f"{hyperopt_config_fly['exp']}.config.json", ".")[0]
# rand_best = research(objective_given_graph,  dataset, f"{hyperopt_config_rand['exp']}.config.json", ".")[0]
print(fly_best)
print(rand_best)


#======================================================================
timesteps = 1500
# x0 = [0.37926545, 0.058339, -0.08167691]
# X = doublescroll(timesteps, x0=x0, method="RK23")
X = mackey_glass(timesteps)# lorenz(timesteps)

train_len = 900
X_train = X[:train_len]
y_train = X[1 : train_len + 1]
X_test = X[train_len : -1]
y_test = X[train_len + 1:]
dataset = ((X_train, y_train), (X_test, y_test))


#======================================================================
warmup = 100

rand_model = get_ESN(create_random(load_fly()), rand_best, use_defaults=True)
rand_model = rand_model.fit(X_train, y_train)
warmup_y = rand_model.run(X_train[:warmup], reset=True)

rand_predictions = np.empty((timesteps - warmup, 1))
x = warmup_y[-1].reshape(1, -1)

for i in tqdm(range(timesteps - warmup)):
    x = rand_model(x)
    rand_predictions[i] = x
#
# .run(X)

fly_model = get_ESN(load_fly(), fly_best, use_defaults=True)
fly_model = fly_model.fit(X_train, y_train)

warmup_y = fly_model.run(X_train[:warmup], reset=True)

fly_predictions = np.empty((timesteps - warmup, 1))
x = warmup_y[-1].reshape(1, -1)

for i in tqdm(range(timesteps - warmup)):
    x = fly_model(x)
    fly_predictions[i] = x

    # .run(X)

print(f"MSE fly: {np.mean(np.sum((X[warmup:] - fly_predictions) ** 2, axis=1))}")
print(f"MSE rand: {np.mean(np.sum((X[warmup:] - rand_predictions) ** 2, axis=1))}")

plt.plot(np.sum((X[warmup:] - fly_predictions) ** 2, axis=1), label = "Point Squared Error Fly")
plt.plot(np.sum((X[warmup:] - rand_predictions) ** 2, axis=1), label = "Point Squared Error Random")
plt.yscale('log')
plt.legend()
plt.show()
plt.close()




fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111)
ax.set_title("Mackey glass")
ax.set_xlabel("x")
ax.set_ylabel("y")
# # ax.set_zlabel("z")
ax.grid(False)


ax.plot(X[warmup:], label = "real")
ax.plot(fly_predictions, label = "fly")
ax.plot(rand_predictions, label = "rand")

ax.legend(shadow=True)

plt.show()



#
#
# for i in range(warmup, timesteps-1):
#     if i == timesteps - 2:
#         ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color = "cyan", label = "Original Data", lw = 0.)
#     else:
#         ax.plot(X[i:i + 2, 0], X[i:i + 2, 1], X[i:i + 2, 2], color="cyan",
#                 alpha=0.75 + 0.25 * (i / (timesteps - 1)), lw=0.4)
#
# for i in range(warmup, timesteps-1):
#     if i == timesteps - 2:
#         ax.plot(fly_predictions[i:i+2, 0], fly_predictions[i:i+2, 1], fly_predictions[i:i+2, 2], color = "Orange", label = "Fly Brain", lw=0.)
#     else:
#         ax.plot(fly_predictions[i:i + 2, 0], fly_predictions[i:i + 2, 1], fly_predictions[i:i + 2, 2],
#                 color="orange", alpha=0.75 + 0.25 * (i / (timesteps - 1)), lw=0.4)
#
# for i in range(warmup, timesteps-1):
#     if i == timesteps - 2:
#         ax.plot(rand_predictions[i:i+2, 0], rand_predictions[i:i+2, 1], rand_predictions[i:i+2, 2], color = "green", label = "Random", lw=0.)
#     else:
#         ax.plot(rand_predictions[i:i + 2, 0], rand_predictions[i:i + 2, 1], rand_predictions[i:i + 2, 2],
#                 color="green", alpha=0.75 + 0.25 * (i / (timesteps - 1)), lw=0.4)
#
#
# ax.set_xlim([np.max(X[:,0]), np.min(X[:,0])])
# ax.set_ylim([np.max(X[:,1]), np.min(X[:,1])])
# ax.set_zlim([np.max(X[:,2]), np.min(X[:,2])])
#
# ax = plt.gca()
# ax.set_facecolor((0, 0, 0))
# ax.axis('off')
# fig.patch.set_facecolor('black')
# plt.tight_layout(w_pad=0, h_pad=0)
# plt.savefig("Lorenz-attractor.png", dpi=800)

# plt.show()



