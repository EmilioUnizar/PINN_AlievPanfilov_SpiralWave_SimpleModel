import torch
import numpy as np
import wandb
from src.solver.solver import Solver
import random
import matplotlib
import datetime  # Import datetime module
import json  # Import json for parsing hidden_dim if needed
matplotlib.use('Agg')

torch.set_float32_matmul_precision('medium')
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(config=None):
    # Get current datetime for the run name
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize W&B run
    run = wandb.init(config=config)
    args = wandb.config

    # Set all random seeds for complete reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create the run name with datetime and hyperparameters
    name = f"{current_time}_ph{args.factor_ph:.2e}_ic{args.factor_ic:.2e}_bc{args.factor_bc:.2e}"
    print(f"Run name: {name}")
    args.wandb_name = name
    args.wandb_project = "ignored"  # W&B project just for consistency, its ignored in the sweep
    
    # Set the run name in W&B
    if run is not None:
        run.name = name

    # Initialize the solver with the name
    solver = Solver(args)
    solver.train()
    solver.test()


if __name__ == '__main__':
    train()