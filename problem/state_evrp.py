import torch
from typing import NamedTuple
from .boolmask import mask_long2bool, mask_long_scatter


class StateEVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor
    types: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    dist_matrix: torch.Tensor

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    used_battery: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    min_battery: torch.Tensor

    VEHICLE_CAPACITY = 1.0  # Hardcoded
    VEHICLE_BATTERY = 1.0
    battery_use_coef = 0.3


    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __my_getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            min_battery=self.min_battery[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot'] if len(input['depot'].shape) == 3 else input['depot'].unsqueeze(1)
        loc = input['loc']
        demand = input['demand']
        types = input['types']
        all_nodes = torch.cat((depot, loc), dim=1)
        dist_matrix = (all_nodes[:, :, None, :] - all_nodes[:, None, :, :]).norm(p=2, dim=-1)

        # Compute min_battery once
        RS_idx = (all_nodes.shape[1] - demand.shape[1])
        RS_coords = all_nodes[:, :RS_idx, :].unsqueeze(2)
        Customer_coords = all_nodes.unsqueeze(1)
        min_battery, _ = torch.min(0.1 * torch.sqrt(torch.sum((RS_coords - Customer_coords) ** 2, dim=-1)), dim=1)

        batch_size, n_loc, _ = loc.size()
        return StateEVRP(
            coords=torch.cat((depot, loc), -2),
            demand=demand,
            types = types,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            used_battery=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'],  # Add step dimension
            dist_matrix=dist_matrix.to(all_nodes.device),
            min_battery=min_battery.to(all_nodes.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if state represents single step"
        # Update the state
        selected = selected[:, None]  # Add dimension for step
        starts = self.prev_a
        prev_a = selected
        demands = self.demand.squeeze(-1)
        n_loc = demands.size(-1)  # Excludes depot
        nodes_num = self.coords.shape[1]
        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))

        all_demand = torch.cat((torch.zeros(self.coords.shape[0], self.coords.shape[1] - self.demand.shape[1]).to(demands.device), demands), dim=1)
        selected_demand = all_demand[self.ids, prev_a]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        selected_battery = self.battery_use_coef * self.dist_matrix[self.ids.squeeze(), starts.squeeze(),selected.squeeze()].unsqueeze(1)
        used_battery = (self.used_battery + selected_battery) * (prev_a >= (nodes_num - n_loc)).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)
        # print(visited_)
        # breakpoint()
        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, used_battery=used_battery, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        customer_idx_start = self.coords.shape[1] - self.demand.shape[1]
        return (self.visited[:, :, customer_idx_start:] != 0).all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_current_time(self):
        return self.prev_a.new_zeros(self.prev_a.shape[0], 1, 1, dtype=torch.float32)
    
    def get_minbattery_back_to_RS(self):
        RS_idx = (self.coords.shape[1] - self.demand.shape[1])
        RS_coords = self.coords[:,:RS_idx, :].unsqueeze(2)
        Customer_coords = self.coords.unsqueeze(1)
        min_battery, _ = torch.min(self.battery_use_coef * torch.sqrt(torch.sum((RS_coords - Customer_coords) ** 2, dim=-1)), dim=1) # shape: [bs, RS_idx, n-RS_idx]
        return min_battery.to(self.coords.device) 

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids visiting the depot twice in a row unless all nodes have been visited.
        :return: A feasibility mask tensor.
        """
        rs_idx = self.coords.shape[1] - self.demand.shape[1]
        rs_number = rs_idx - 1
        batch_size = self.coords.shape[0]
        device = self.visited_.device

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:].clone()
            # Reset RS visits
            visited_loc[:, :, :rs_number] = 0
            # Prevent RS revisits directly after RS visit
            rs_mask = (self.prev_a.unsqueeze(-1) < rs_idx) & (self.prev_a.unsqueeze(-1) > 0)
            visited_loc[:, :, :rs_idx] |= rs_mask.expand_as(visited_loc[:, :, :rs_idx])
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        prev_loc = self.prev_a.squeeze()
        demands = torch.cat((torch.zeros(batch_size, rs_idx, 1, device=device), self.demand), dim=1)
        exceeds_cap = (demands[self.ids].squeeze(-1) + self.used_capacity[:, :, None]) > self.VEHICLE_CAPACITY

        # Calculate battery exceed condition
        dist_to_next = self.dist_matrix[self.ids.squeeze(), prev_loc, :].unsqueeze(1)
        battery_usage = self.battery_use_coef * dist_to_next
        exceeds_battery = (battery_usage + self.used_battery[:, :, None] + self.min_battery.unsqueeze(1)) > self.VEHICLE_BATTERY

        # Depot visit mask (cannot revisit depot unless all nodes are served)
        mask_depot = (self.prev_a == 0).unsqueeze(-1)
        new_visit = torch.cat((mask_depot, visited_loc), dim=-1)

        # print(new_visit[0], exceeds_cap[0], exceeds_battery[0])
        mask_loc = new_visit | exceeds_cap | exceeds_battery
        # Check unvisited customer nodes to enforce depot revisiting rule
        all_customers_visited = (new_visit[:, :, rs_idx:] != 0).all(dim=-1, keepdim=True).squeeze()
        mask_loc[all_customers_visited, :, :] = True
        mask_loc[all_customers_visited, :, 0] = False

        # Prevent revisiting RS immediately after depot visit
        start_at_depot = (self.get_current_node() == 0).squeeze()
        mask_loc[start_at_depot, :, :rs_idx] = True

        avoid_nan_mask = (torch.sum(mask_loc, dim=2) == mask_loc.shape[-1]).squeeze()
        mask_loc[avoid_nan_mask, :, 0] = False
        return mask_loc.to(dtype=torch.bool)

    def get_mask_(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        rs_idx = self.coords.shape[1] - self.demand.shape[1]
        rs_number = rs_idx - 1
        batch_size = self.coords.shape[0]
        if self.visited_.dtype == torch.uint8:
            # RS + Customer Nodes
            visited_loc = self.visited_[:, :, 1:]
            # RS Can be visited many times (RS idx - 1, since we start from 1)
            visited_loc[:,:,:rs_number] = 0
            # If we start from a RS, it cannot visit itself
            for i in range(batch_size):
                if 0 < self.prev_a[i] < rs_idx:
                    visited_loc[i, :, :rs_idx] = 1

        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # print(self.prev_a, visited_loc)
        # breakpoint()
        prev_loc = self.prev_a.squeeze()
        demands = torch.cat((torch.zeros(batch_size, rs_idx, 1).to(self.visited_.device), self.demand),dim=1).to(self.visited_.device)
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = demands[self.ids].squeeze(-1) + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY
        # Nodes that cannot be visited are already visited or too much demand to be served now
        # exceed_battery
        exceeds_battery = ( (self.battery_use_coef * self.dist_matrix[self.ids.squeeze(),prev_loc,:].unsqueeze(1) + self.used_battery[:, :, None] + self.min_battery.unsqueeze(1)) > self.VEHICLE_BATTERY)

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0)
        new_visit = torch.cat((mask_depot[:,:,None], visited_loc), dim=-1)
        mask_loc = new_visit.to(exceeds_cap.dtype) | exceeds_cap | exceeds_battery
        # print(new_visit[0], exceeds_cap[0], exceeds_battery[0])
        for i in range(batch_size):
            if (new_visit[i,:,rs_idx:] != 0).all():   
                mask_loc[i,:,:] = True
                mask_loc[i,:,0] = False
            else:
                if self.get_current_node()[i] == 0:
                    mask_loc[i,:,:rs_idx] = True
        # Visit a customer node, next node has no depot / RS, set it as False
        return mask_loc.to(dtype=torch.bool)

    def construct_solutions(self, actions):
        return actions
