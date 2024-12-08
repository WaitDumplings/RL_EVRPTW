from torch.utils.data import Dataset
import torch
import os
import pickle
import random
import numpy as np

from .state_evrp import StateEVRP

def count_routes_per_row(row):
    """count routes"""
    routes = 0
    in_route = False

    for value in row:
        if value != 0 and not in_route:
            # start a new route
            in_route = True
        elif value == 0 and in_route:
            # complete a route
            routes += 1
            in_route = False

    return routes


class EVRP(object):

    NAME = 'evrp'  # Capacitated Vehicle Routing Problem
    BATTERY_CAPACITY = 1.0
    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, remain_energy, pi, remain_coef = 3, dis_loss_coef = 0.99, tra_len_coef = 0.01, route_fee=3):
        my_demand = torch.cat((torch.zeros(dataset['demand'].shape[0], dataset['loc'].shape[1] - dataset['demand'].shape[1], 1).to(pi.device), dataset['demand']),dim=1)
        # rs_idx = list(range(1, dataset['loc'].shape[1] - dataset['demand'].shape[1] + 1))

        # Traject length
        reverse_data = pi.data.flip(dims=[1])

        # number of routes
        total_routes = torch.from_numpy(np.array([count_routes_per_row(row) for row in reverse_data])).to(pi.device)

        # ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(my_demand[:, :1], -EVRP.VEHICLE_CAPACITY),
                my_demand
            ),
            1
        ).squeeze(-1)

        pi = torch.cat((torch.zeros(pi.shape[0], 1).to(pi.device).to(torch.int64), pi), dim=1)
        d = demand_with_depot.gather(1, pi)
        used_cap = torch.zeros_like(my_demand.squeeze(-1)[:, 0])

        for i in range(pi.size(1)):
            used_cap += d[:, i] # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            if not (used_cap <= EVRP.VEHICLE_CAPACITY + 1e-5).all():
                breakpoint()
            assert (used_cap <= EVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        # breakpoint()
        loc_with_depot = torch.cat((dataset['depot'], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        total_dis = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot'].squeeze(1)).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot'].squeeze(1)).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        )
        cost1 = dis_loss_coef * total_dis 
        cost2 = route_fee * total_routes
        cost3 = remain_energy.squeeze() * remain_coef
        dis_loss = cost1 + cost2 + cost3
        
        # print("Cost Ratio is {} - {} - {}".format(cost1[0], cost2[0], cost3[0]))
        return dis_loss, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateEVRP.initialize(*args, **kwargs)

def make_instance(args):
    depot, loc, demand, types, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'types': torch.tensor(types, dtype=torch.int64)
    }


class VRPDataset(Dataset):
    def __init__(self, filename=None, size=50, ratio = 0.9, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.customer_number = int(ratio * size)
            self.rs_number = size - self.customer_number
            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(self.customer_number).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    'types': torch.cat((torch.zeros(self.rs_number,1), torch.ones(self.customer_number,1)))
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

