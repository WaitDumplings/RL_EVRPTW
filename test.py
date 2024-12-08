import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from model import EVRP_Model
from configuration import Config
from problem.evrp import EVRP


def parse_arguments():
    parser = argparse.ArgumentParser(description="EVRP Model Runner")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--output_dir", type=str, default="images", help="Directory to save output images.")
    parser.add_argument("--graph_size", type=int, default=50, help="Size of the EVRP graph.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples in the dataset.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for the dataloader.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    return parser.parse_args()


def load_model(path, config, epoch=None):
    if os.path.isfile(path):
        model_filename = path
    else:
        raise ValueError(f"{path} is not a valid file path")

    problem = EVRP()
    model = EVRP_Model(config, problem=problem)

    load_data = torch.load(model_filename, map_location=lambda storage, loc: storage)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model.eval()
    return model


def plot_vehicle_routes(data, route, ax1, visualize_demands=False, demand_scale=1, round_demand=False, markersize=5):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    # route is one sequence, separating different routes with 0 (depot)
    battery_capacity = 1.0
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


def main():
    args = parse_arguments()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load model
    config = Config()
    model = load_model(args.model_path, config=config)

    # Prepare dataset and dataloader
    dataset = EVRP.make_dataset(size=args.graph_size, num_samples=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Evaluate model
    batch = next(iter(dataloader))
    model.eval()
    model.decoder.decode_type = "greedy"

    with torch.no_grad():
        res, likelihood, pi = model(batch, return_pi=True)

    # Plot and save results
    os.makedirs(args.output_dir, exist_ok=True)
    for i, (data, tour) in enumerate(zip(dataset, pi)):
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(data, tour, ax, visualize_demands=False, demand_scale=50, round_demand=True)
        fig.savefig(os.path.join(args.output_dir, f'cvrp_{i}.png'))


if __name__ == "__main__":
    main()
