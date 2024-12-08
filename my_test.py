import numpy as np
import torch
from torch.utils.data import DataLoader
from generate_data import generate_evrp_data
from problem.evrp import EVRP
# %matplotlib inline
from matplotlib import pyplot as plt
import os
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(data, route, ax1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """

    # route is one sequence, separating different routes with 0 (depot)
    battery_capacity = 1.0
    route_fee = 3
    routes = [r[r!=0] for r in np.split(route.cpu().numpy(), np.where(route==0)[0]) if (r != 0).any()]
    depot = data['depot'].cpu().numpy()
    locs = data['loc'].cpu().numpy()
    demands = data['demand'].cpu().numpy() * demand_scale
    demands = np.trunc(demands * 10**4) / 10**4

    rs_number = data['loc'].shape[0] - data['demand'].shape[0]
    demands = np.concatenate((np.zeros(rs_number, dtype=demands.dtype), demands))

    capacity = demand_scale # Capacity is always 1
    
    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)

    # Plot recharging stations (RS) as red points
    rs_coords = locs[:rs_number]
    ax1.plot(rs_coords[:, 0], rs_coords[:, 1], 'ro', markersize=markersize * 3, label='Recharging Station')

    for i, (x, y) in enumerate(rs_coords):
        ax1.annotate(
            'RS', (x, y),
            textcoords="offset points", xytext=(5, 5),
            ha='center', fontsize=12, fontweight='bold'
        )
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    
    legend = ax1.legend(loc='upper center')
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        coords = np.concatenate(
            (np.expand_dims(data['depot'], axis=0), coords, np.expand_dims(data['depot'], axis=0)),
            axis=0
        )
        xs, ys = coords.transpose()
        total_route_demand = sum(route_demands)
        if total_route_demand > capacity:
            breakpoint()
        assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0

        route_demands = np.concatenate((np.zeros(1), route_demands, np.zeros(1)))
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            # cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            # used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            # dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            
            x_prev, y_prev = x, y
            cum_demand += d
        # if dist * 0.3 > 1:
        #     breakpoint()
        # print(dist * 0.3)
  
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        total_dist += len(routes) * route_fee
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                int(total_route_demand) if round_demand else total_route_demand, 
                int(capacity) if round_demand else capacity,
                dist
            )
        )
        
        qvs.append(qv)

    ax1.set_title('{} routes, total distance {:.2f}'.format(len(routes), total_dist))
    ax1.legend(handles=qvs)
    
    # pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    # pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    # pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
    
    # if visualize_demands:
    #     ax1.add_collection(pc_cap)
    #     ax1.add_collection(pc_used)
    #     ax1.add_collection(pc_dem)

import json
from model import EVRP_Model
from configuration import get_options, Config

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args

def load_model(path, config, epoch=None):

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = EVRP()

    model =  EVRP_Model(config, problem = problem)
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args

from model import EVRP_Model
from configuration import get_options, Config
# Figure out what's the problem
problem = EVRP()

# Initialize model
# config = Config()
# model = EVRP_Model(config, problem = problem)

model_path = '/data/RL_EVRPTW/outputs/evrp_50/run_20241207T163502/epoch-0.pt'
model, _= load_model(model_path, config=Config())

torch.manual_seed(1234)
dataset = EVRP.make_dataset(size=50, num_samples=10)

# Need a dataloader to batch instances
dataloader = DataLoader(dataset, batch_size=5)

# Make var works for dicts
batch = next(iter(dataloader))

# Run the model
model.eval()
model.decoder.decode_type = "greedy"
with torch.no_grad():
    import time
    tic = time.time()
    res, likelihood, pi = model(batch, return_pi=True)
    print( (time.time() - tic)/5, "S")
tours = pi
# Plot the results
for i, (data, tour) in enumerate(zip(dataset, tours)):
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_vehicle_routes(data, tour, ax, visualize_demands=False, demand_scale=50, round_demand=True)
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(os.path.join('images', 'cvrp_{}.png'.format(i)))
