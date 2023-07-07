import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir

def random_weights(graph):
    random_assignment = np.random.random(graph.shape)
    return graph * random_assignment

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
    #
    # random_assignment = np.random.random(fly_graph.shape)
    # fly_graph *= random_assignment

    if return_tensor:
        return torch.Tensor(fly_graph)
    else:
        return fly_graph

def create_random(fly_graph, return_tensor = False):
    rand_graph = nx.fast_gnp_random_graph(fly_graph.shape[0], np.sum(fly_graph) / (fly_graph.shape[0] ** 2))
    rand_graph = nx.to_numpy_array(rand_graph)
    #
    # random_assignment = np.random.random(rand_graph.shape)
    # rand_graph *= random_assignment

    if return_tensor:
        return torch.Tensor(rand_graph)
    else:
        return rand_graph

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
                              inut_scaling=iss,
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
    if graph_to_use == "fly":
        graph = load_fly()
    else:
        graph = create_random(load_fly())

    losses = []; r2s = [];
    for n in range(instances):
        # Build your model given the input parameters
        reservoir_graph = random_weights(graph)
        reservoir = Reservoir(W = reservoir_graph,
                              sr=sr,
                              lr=lr,
                              inut_scaling=iss,
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

# if __name__ == "__main__":
# fly_graph = load_fly()
# rand_graph = create_random(fly_graph)
#
# fly_reservoir = Reservoir(W = fly_graph)
# rand_reservoir    = Reservoir(W = rand_graph)
#
from reservoirpy.nodes import Reservoir, Ridge
#
# # reservoir = Reservoir(100, lr=0.5, sr=0.9)
# ridge_fly = Ridge(ridge=1e-7)
# ridge_rand = Ridge(ridge=1e-7)
#
# esn_fly = fly_reservoir >> ridge_fly
# esn_rand = rand_reservoir >> ridge_rand

# X = np.sin(np.linspace(0, 6*np.pi, 300)).reshape(-1, 1)
# X_train = X[:149]
# Y_train = X[1:150]
#
# esn_fly  = esn_fly.fit(X_train, Y_train, warmup=10)
# esn_rand = esn_rand.fit(X_train, Y_train, warmup=10)
#
# import matplotlib.pyplot as plt
#
# Y_pred_fly = esn_fly.run(X[150:])
# Y_pred_rand = esn_rand.run(X[150:])
#
# plt.figure(figsize=(10, 3))
# plt.title("A sine wave and its future.")
# plt.xlabel("$t$")
# plt.plot(Y_pred_fly, label="Predicted sin(t+1) Fly", color="green")
# plt.plot(Y_pred_rand, label="Predicted sin(t+1) Rand", color="blue")
# plt.plot(X[150:], label="Real sin(t+1)", color="red")
# plt.legend()
# plt.show()

from reservoirpy.datasets import doublescroll, lorenz, multiscroll
from reservoirpy.observables import nrmse, rsquare


# params := alpha = 10.82, beta = 14.286, a = 1.3, b = .11, c = 7, d = 0
#
# initv := x(0) = 1, y(0) = 1, z(0) = 0

timesteps = 15000
# x0 = [0.37926545, 0.058339, -0.08167691]
# X = doublescroll(timesteps, x0=x0, method="RK23")
X = lorenz(timesteps, h=0.01)
# X = multiscroll(timesteps)
# fig = plt.figure(figsize=(10, 10))
# ax  = fig.add_subplot(111, projection='3d')
# ax.set_title("Double scroll attractor (1998)")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.grid(False)
#
# for i in range(timesteps-1):
#     ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color=plt.cm.cividis(255*i//timesteps), lw=1.0)
#
# plt.show()


hyperopt_config_fly = {
    "exp": f"hyperopt-multiscroll-fly", # the experimentation name
    "hp_max_evals": 50,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "graph_to_use": ["choice", "fly"],
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-6 and 10
        "lr": ["loguniform", 1e-3, 1],    # idem with the leaking rate, from 1e-3 to 1
        "iss": ["uniform", 0.1, 0.9],           # the input scaling is fixed
        "ridge": ["choice", 1e-7],        # and so is the regularization parameter.
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
        "ridge": ["choice", 1e-7],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}


import json
from reservoirpy.hyper import research
from reservoirpy.hyper import plot_hyperopt_report

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config_fly['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config_fly, f)

with open(f"{hyperopt_config_rand['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config_rand, f)

train_len = 12000
X_train = X[:train_len]
y_train = X[1 : train_len + 1]
X_test = X[train_len : -1]
y_test = X[train_len + 1:]
dataset = ((X_train, y_train), (X_test, y_test))

fly_best = {'input_scaling': 0.5872833192706555, 'lr': 0.15492403550483738, 'sr': 0.07071032186253305}
rand_best = {'input_scaling': 0.723792790570117, 'lr': 0.0236187254130727, 'sr': 0.0014791230744112718}

fly_graph = load_fly()
rand_graph = create_random(fly_graph)

fly_reservoir = random_weights(fly_graph)
fly_reservoir = Reservoir(W=fly_reservoir,**fly_best)

fly_readout = Ridge(ridge=1e-7)
fly_model = fly_reservoir >> fly_readout

# Train your model and test your model.
fly_reservoir = fly_model.fit(X_train, y_train)
fly_predictions = fly_reservoir.run(X)

rand_reservoir = random_weights(rand_graph)
rand_reservoir = Reservoir(W=rand_reservoir,**rand_best)

rand_readout = Ridge(ridge=1e-7)
rand_model = rand_reservoir >> rand_readout

# Train your model and test your model.
rand_reservoir = rand_model.fit(X_train, y_train)
rand_predictions = rand_reservoir.run(X)


print(f"MSE fly: {np.mean(np.sum((X - fly_predictions) ** 2, axis=1))}")
print(f"MSE rand: {np.mean(np.sum((X - rand_predictions) ** 2, axis=1))}")

plt.plot(np.sum((X - fly_predictions) ** 2, axis=1), label = "Point Squared Error Fly")
plt.plot(np.sum((X - rand_predictions) ** 2, axis=1), label = "Point Squared Error Random")
plt.yscale('log')
plt.legend()
plt.show()

# timesteps = 200
# x0 = [0.37926545, 0.058339, -0.08167691]
# X = doublescroll(timesteps, x0=x0, method="RK23")
# X = lorenz(timesteps, x0=x0)

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("Lorenz attractor")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.grid(False)

# ax.plot(X[:, 0], X[:, 1], X[:, 2], label = "real", lw=1., alpha = 0.75, c = "black")
# ax.plot(fly_predictions[:, 0], fly_predictions[:, 1], fly_predictions[:, 2], label = "Fly", lw=1., alpha = 0.75, c = "green")
# ax.plot(rand_predictions[:, 0], rand_predictions[:, 1], rand_predictions[:, 2], label = "Random", lw=1., alpha = 0.75, c = "red")
# ax.legend(shadow=True)

# ax.set_xlim([-30,30])
# ax.set_ylim([-30,30])
# ax.set_zlim([0,30])

for i in range(500, timesteps-1):
    if i == timesteps - 2:
        ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color = "Blue", label = "Original Data", lw = 0.)
    else:
        # ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color = plt.cm.viridis(i  / (timesteps - 1)), alpha = 0.5 + 0.5 * (i  / (timesteps - 1)), lw=0.2)
        ax.plot(X[i:i + 2, 0], X[i:i + 2, 1], X[i:i + 2, 2], color="blue",
                alpha=0.5 + 0.5 * (i / (timesteps - 1)), lw=0.2)
        
for i in range(500, timesteps-1):
    if i == timesteps - 2:
        ax.plot(fly_predictions[i:i+2, 0], fly_predictions[i:i+2, 1], fly_predictions[i:i+2, 2], color = "Orange", label = "Fly Brain", lw=0.)
    else:
        # ax.plot(fly_predictions[i:i+2, 0], fly_predictions[i:i+2, 1], fly_predictions[i:i+2, 2], color =plt.cm.inferno(i  / (timesteps - 1)), alpha = 0.5 + 0.5 * (i  / (timesteps - 1)), lw=0.2)
        ax.plot(fly_predictions[i:i + 2, 0], fly_predictions[i:i + 2, 1], fly_predictions[i:i + 2, 2],
                color="orange", alpha=0.5 + 0.5 * (i / (timesteps - 1)), lw=0.2)


ax.set_xlim([np.max(X[:,0]), np.min(X[:,0])])
ax.set_ylim([np.max(X[:,1]), np.min(X[:,1])])
ax.set_zlim([np.max(X[:,2]), np.min(X[:,2])])

#
# for i in range(timesteps-1):
#     # if i == 0:
#     ax.plot(rand_predictions[i:i+2, 0], rand_predictions[i:i+2, 1], rand_predictions[i:i+2, 2], color="green", lw=1.0, alpha = (i  / (timesteps - 1))**2)
ax = plt.gca()
# ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((0, 0, 0))
ax.axis('off')
fig.patch.set_facecolor('black')
# ax.legend()
plt.tight_layout(w_pad=0, h_pad=0)
plt.savefig("Lorenz-attractor.png", dpi=800)

plt.show()

# best_fly = research(objective_given_graph,  dataset, f"{hyperopt_config_fly['exp']}.config.json", ".")
#
#
# # fig = plot_hyperopt_report(hyperopt_config_fly["exp"], ("lr", "sr"), metric="r2")
# # plt.show()
#
# best_rand = research(objective_given_graph,  dataset, f"{hyperopt_config_rand['exp']}.config.json", ".")
#
# print(f"FLY BEST: {best_fly}")
# print(f"RANDOM BEST: {best_rand}")



# fig = plot_hyperopt_report(hyperopt_config_rand["exp"], ("lr", "sr"), metric="r2")
# plt.show()





