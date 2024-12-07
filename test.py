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


def plot_vehicle_routes(data, route, ax1, visualize_demands=False, demand_scale=1, round_demand=False):
    # Simplified plotting code here
    pass


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
