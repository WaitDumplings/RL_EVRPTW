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
    def get_costs(dataset, pi, rs_loss_coef = 0.01, dis_loss_coef = 0.99, tra_len_coef = 0.01, route_fee=5):
        my_demand = torch.cat((torch.zeros(dataset['demand'].shape[0], dataset['loc'].shape[1] - dataset['demand'].shape[1], 1).to(pi.device), dataset['demand']),dim=1)
        rs_idx = list(range(1, dataset['loc'].shape[1] - dataset['demand'].shape[1] + 1))
        rs_idx_tensor = torch.tensor(rs_idx, device=pi.device)

        batch_size, graph_size, _ = my_demand.size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]
        counts = (sorted_pi.unsqueeze(-1) == rs_idx_tensor).view(batch_size, -1).sum(dim=1)
        
        # Traject length
        reverse_data = pi.data.flip(dims=[1])
        non_zero_idx = (reverse_data != 0).float().argmax(dim=1)  # find the first non zero
        real_lengths = pi.data.size(1) - non_zero_idx

        # number of routes
        total_routes = torch.from_numpy(np.array([count_routes_per_row(row) for row in reverse_data])).to(pi.device)

        # Sorting it should give all zeros at front and then 1...n
        # assert (
        #     torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
        #     sorted_pi[:, -graph_size:]
        # ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(my_demand[:, :1], -EVRP.VEHICLE_CAPACITY),
                my_demand
            ),
            1
        ).squeeze(-1)

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
        loc_with_depot = torch.cat((dataset['depot'], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        total_dis = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot'].squeeze(1)).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot'].squeeze(1)).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        )
        rs_loss = rs_loss_coef * counts
        dis_loss = dis_loss_coef * total_dis + real_lengths * tra_len_coef + route_fee * total_routes

        return dis_loss + rs_loss, None

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

