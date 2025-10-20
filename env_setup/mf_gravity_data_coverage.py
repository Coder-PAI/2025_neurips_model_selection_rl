import argparse
import os
from typing import List
import numpy as np


def create_grid(center, gap, size):
    assert size % 2 == 1
    return [center + gap * i for i in range(-(size // 2), size // 2 + 1)]


def merge_grids(grids: List[List[float]]):
    centers = set(grid[len(grid) // 2] for grid in grids)
    assert len(centers) == 1
    merged_grids = []
    for grid in grids:
        print(grid)
        merged_grids += grid
    merged_grids = sorted(list(set(merged_grids)))
    indices = {g: index for index, g in enumerate(merged_grids)}
    return merged_grids, indices


def create_grids_to_one(center, gaps, size):
    grids = [create_grid(center, gap, size) for gap in gaps]
    return merge_grids(grids)


def fetch_grid(indices, center, gap, size):
    return list(indices[center + gap * i] for i in range(-(size // 2), size // 2 + 1))


def fetch_center(grid):
    return len(grid) // 2


gravity_grid = [-30 + 3 * i for i in range(-7, 8)]
noise_grid = [100.0]
policy_list = [8]
g_size = len(gravity_grid)
n_size = len(noise_grid)
grid_size = g_size * n_size
dataset_list = [0, grid_size // 2, grid_size - 1]

params = [(gravity, noise) for gravity in gravity_grid for noise in noise_grid]
gravity_list, noise_list = zip(*params)

completed_policies = []
text_for_comparison = ""


class ModelSelectionArgs:
    def __init__(self):
        self.args = {
            # Embarrassingly Parallelism for Rollout Simulations
            "sessions": 24,
            "parallel_terminals_on": False,
            "enable_embarrassingly": True,
            "jump_to_validation": False,
            "enable_bellman_operators": False,
            # MDP Settings
            "gamma": 0.99,
            "hopper_gravities": gravity_list,
            "hopper_noise_scaler_list": noise_list,
            "hopper_noise_mean_list": [0 for _ in range(grid_size)],
            # PolicyTrainer Settings
            "trainer_epsilon": 0.3,
            "trainer_total_steps": 100000,
            "trainer_checkpoint": 1,
            "lr": 3e-5,
            "trainer_limit": 1500000,
            "hidden_layers": [512, 256],
            "trainer_algorithm": "DDPG",
            "clear_ckpt_after_train": True,
            # OfflineDatasetCollector Settings
            "sampler_epsilon": 0.3,
            "sampler_max_horizon": 1024,
            "sampler_size": 3200,
            "sampler_noise_scaler": 1.0,
            "sampler_verbose": True,
            # RolloutEvaluator Settings
            "rollout_max_horizon": 1024,
            "rollout_num_trajectories": 128,
            "rollout_batch_size": 64,
            "placeholder": "placeholder.txt",
            # Selector Settings
            "model_based_batch_size": 32,
            "model_based_state_norm": 1,
            "model_based_sample_norm": 2,
            "model_based_obs_loss_order": 1,
            "model_based_compare_pred_truth": False,
            # Validation Settings
            "misclassified": True,
            "top_k": 1,
            "request_interval": 10,
            "policy_for_traverse": policy_list,
            "ground_truth_for_traverse": dataset_list,
        }

    @staticmethod
    def str2bool(arg: str):
        assert arg.upper() in ["TRUE", "FALSE"]
        return True if arg.upper() == "TRUE" else False

    @staticmethod
    def str2list_int(arg: str):
        return [int(x) for x in arg.split(",")]


def get_arguments(_args):
    # print("\n" + Fore.YELLOW + "Arguments in the model selection experiment..." + Style.RESET_ALL)
    logging_info = "".join([f"--{arg}={value},\n" for arg, value in vars(_args).items()])
    os.makedirs(f"offline_data", exist_ok=True)
    with open(f"offline_data/experiment_setup.txt", "w") as file:
        file.write(logging_info)
    # print(logging_info)
    # print("\n\n")


def parse_args():
    default = ModelSelectionArgs()
    parser = argparse.ArgumentParser(description="Model Selection Arguments")

    # MDP Settings
    parser.add_argument("--gamma", type=float, default=default.args["gamma"])
    parser.add_argument(
        "--hopper_gravities",
        nargs="+",
        type=int,
        default=default.args["hopper_gravities"],
    )
    parser.add_argument(
        "--hopper_noise_scaler_list",
        nargs="+",
        type=float,
        default=default.args["hopper_noise_scaler_list"],
    )
    parser.add_argument(
        "--hopper_noise_mean_list",
        nargs="+",
        type=float,
        default=default.args["hopper_noise_mean_list"],
    )
    # PolicyTrainer Settings
    parser.add_argument(
        "--trainer_epsilon",
        type=float,
        default=default.args["trainer_epsilon"],
    )
    parser.add_argument(
        "--trainer_total_steps",
        type=int,
        default=default.args["trainer_total_steps"],
    )
    parser.add_argument(
        "--trainer_checkpoint",
        type=int,
        default=default.args["trainer_checkpoint"],
    )
    parser.add_argument("--lr", type=float, default=default.args["lr"])
    parser.add_argument(
        "--trainer_limit",
        type=int,
        default=default.args["trainer_limit"],
    )
    parser.add_argument(
        "--hidden_layers",
        nargs="+",
        type=int,
        default=default.args["hidden_layers"],
    )
    parser.add_argument(
        "--trainer_algorithm",
        type=str,
        default=default.args["trainer_algorithm"],
    )
    parser.add_argument(
        "--clear_ckpt_after_train",
        type=default.str2bool,
        default=default.args["clear_ckpt_after_train"],
    )

    # OfflineDatasetCollector Settings
    parser.add_argument(
        "--sampler_epsilon",
        type=float,
        default=default.args["sampler_epsilon"],
    )
    parser.add_argument(
        "--sampler_max_horizon",
        type=int,
        default=default.args["sampler_max_horizon"],
    )
    parser.add_argument(
        "--sampler_size",
        type=int,
        default=default.args["sampler_size"],
    )
    parser.add_argument(
        "--sampler_noise_scaler",
        type=float,
        default=default.args["sampler_noise_scaler"],
    )
    parser.add_argument(
        "--sampler_verbose",
        type=default.str2bool,
        default=default.args["sampler_verbose"],
    )

    # RolloutEvaluator Settings
    parser.add_argument(
        "--rollout_max_horizon",
        type=int,
        default=default.args["rollout_max_horizon"],
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=default.args["rollout_batch_size"],
    )
    parser.add_argument(
        "--rollout_num_trajectories",
        type=int,
        default=default.args["rollout_num_trajectories"],
    )

    # Selector Settings
    parser.add_argument(
        "--model_based_batch_size",
        type=int,
        default=default.args["model_based_batch_size"],
    )
    parser.add_argument(
        "--model_based_state_norm",
        type=int,
        default=default.args["model_based_state_norm"],
    )
    parser.add_argument(
        "--model_based_sample_norm",
        type=int,
        default=default.args["model_based_sample_norm"],
    )
    parser.add_argument(
        "--model_based_obs_loss_order",
        type=int,
        default=default.args["model_based_obs_loss_order"],
    )
    parser.add_argument(
        "--model_based_compare_pred_truth",
        type=default.str2bool,
        default=default.args["model_based_compare_pred_truth"],
    )

    parser.add_argument(
        "--misclassified",
        type=default.str2bool,
        default=default.args["misclassified"],
    )
    parser.add_argument(
        "--jump_to_validation",
        type=default.str2bool,
        default=default.args["jump_to_validation"],
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=default.args["top_k"],
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=default.args["sessions"],
    )
    parser.add_argument(
        "--request_interval",
        type=int,
        default=default.args["request_interval"],
    )
    parser.add_argument(
        "--placeholder",
        type=str,
        default=default.args["placeholder"],
    )
    parser.add_argument(
        "--parallel_terminals_on",
        type=default.str2bool,
        default=default.args["parallel_terminals_on"],
    )
    parser.add_argument(
        "--enable_embarrassingly",
        type=default.str2bool,
        default=default.args["enable_embarrassingly"],
    )
    parser.add_argument(
        "--enable_bellman_operators",
        type=default.str2bool,
        default=default.args["enable_bellman_operators"],
    )

    parser.add_argument(
        "--ground_truth_for_traverse",
        nargs="+",
        type=int,
        default=default.args["ground_truth_for_traverse"],
    )
    parser.add_argument(
        "--policy_for_traverse",
        type=default.str2list_int,
        default=default.args["policy_for_traverse"],
    )
    parsed_args = parser.parse_args()
    if (
            not len(parsed_args.hopper_gravities)
                == len(parsed_args.hopper_noise_scaler_list)
                == len(parsed_args.hopper_noise_mean_list)
    ):
        raise AssertionError("Parameters for Hopper-v4 should align with each other!")
    return parsed_args


args = parse_args()
get_arguments(args)

__all__ = ["get_arguments", "parse_args", "args"]
