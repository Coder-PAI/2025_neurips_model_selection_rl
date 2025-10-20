import csv
import math
import multiprocessing
import os
import parser
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import psutil
from ablation_observer import AblationObserver
from selection_algorithms import DiscriminatorsWithDirectInput
from tqdm import tqdm

from global_utils import GeneralUtils, stat_from_seq

env_indices = list(range(len(parser.args.hopper_gravities)))
policy_indices = parser.args.policy_for_traverse
dataset_indices = parser.args.ground_truth_for_traverse
toolbox = GeneralUtils()
markers = "os^v<>p*Hh+xDd|_12348PX.,123"  # used to mark curves


def available_cpu_cores(percent, limit, num_attempt=16, interval=0.1):
    cores = []
    for _ in range(num_attempt):
        cpu_percentages = psutil.cpu_percent(percpu=True)
        available_cores = sum(
            1 for cpu_percent in cpu_percentages if cpu_percent < percent
        )
        cores.append(available_cores)
        time.sleep(interval)
    num_cores = min(max(int(np.mean(cores)), 1), limit)
    print(f"Pool enabled. Using {num_cores} cpu cores...")
    return num_cores


def load_q_function(env_index: int, policy_index: int, dataset_index, indices):
    all_estimates = toolbox.load_from_pkl(
        f"offline_data/q_functions/{env_index}/{policy_index}/{dataset_index}/q_function"
    )
    return [
        q
        for episode in all_estimates
        for q in np.mean(np.array(episode)[:, indices], axis=1, keepdims=False).tolist()
    ]


def load_v_function(env_index: int, policy_index: int, dataset_index, indices):
    all_estimates = toolbox.load_from_pkl(
        f"offline_data/v_functions/{env_index}/{policy_index}/{dataset_index}/v_function"
    )
    return [
        v
        for episode in all_estimates
        for v in np.mean(
            np.array(episode)[1:, indices], axis=1, keepdims=False
        ).tolist()
    ]


def drop_function_mean_estimates(args):
    grid_index, indices, env_index, policy_index, dataset_index = args
    if os.path.exists(
        f"offline_data/multiple_rollouts_cache/v_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}/v_function.pkl"
    ) and os.path.exists(
        f"offline_data/multiple_rollouts_cache/q_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}/q_function.pkl"
    ):
        return
    q_function = load_q_function(env_index, policy_index, dataset_index, indices)
    v_function = load_v_function(env_index, policy_index, dataset_index, indices)
    os.makedirs(
        f"offline_data/multiple_rollouts_cache/q_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}",
        exist_ok=True,
    )
    os.makedirs(
        f"offline_data/multiple_rollouts_cache/v_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}",
        exist_ok=True,
    )
    toolbox.save_as_pkl(
        f"offline_data/multiple_rollouts_cache/q_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}/q_function",
        q_function,
    )
    toolbox.save_as_pkl(
        f"offline_data/multiple_rollouts_cache/v_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}/v_function",
        v_function,
    )


def drop_iid_trials(grid_size):
    total_rollouts = parser.args.rollout_num_trajectories
    indices_all = list(range(total_rollouts))
    subset_sizes = [int(x) for x in np.linspace(2, total_rollouts, grid_size)]
    subset_indices_all = [indices_all[:size] for size in subset_sizes]
    multi_args = [
        (
            grid_index,
            subset_indices_all[grid_index],
            env_index,
            policy_index,
            dataset_index,
        )
        for grid_index in range(grid_size)
        for env_index in env_indices
        for policy_index in policy_indices
        for dataset_index in dataset_indices
    ]
    num_cores = available_cpu_cores(percent=10, limit=24)
    with Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(drop_function_mean_estimates, multi_args),
                total=len(multi_args),
                desc="Caching different trials...",
            )
        )


def post_processed_q_function(grid_index, env_index, policy_index, dataset_index):
    return toolbox.load_from_pkl(
        f"offline_data/multiple_rollouts_cache/q_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}/q_function"
    )


def post_processed_v_function(grid_index, env_index, policy_index, dataset_index):
    return toolbox.load_from_pkl(
        f"offline_data/multiple_rollouts_cache/v_functions/"
        f"{grid_index}/{env_index}/{policy_index}/{dataset_index}/v_function"
    )


# noinspection SpellCheckingInspection
def selection_with_realizability(args):
    algorithm, trial_index, grid_index, policy_index, dataset_index = args
    if os.path.exists(
        f"offline_data/multiple_rollouts_cache/selection_results/"
        f"{trial_index}/{grid_index}/{algorithm}/{policy_index}/{dataset_index}/result.pkl"
    ):
        return toolbox.load_from_pkl(
            f"offline_data/multiple_rollouts_cache/selection_results/"
            f"{trial_index}/{grid_index}/{algorithm}/{policy_index}/{dataset_index}/result",
        )
    indices = list(
        np.random.choice(
            range(parser.args.sampler_size), parser.args.sampler_size, replace=True
        )
    )
    q_cur_tables = np.array(
        [
            post_processed_q_function(
                grid_index, env_index, policy_index, dataset_index
            )
            for env_index in env_indices
        ]
    )[:, indices].tolist()
    v_next_tables = np.array(
        [
            post_processed_v_function(
                grid_index, env_index, policy_index, dataset_index
            )
            for env_index in env_indices
        ]
    )[:, indices].tolist()
    all_rewards = np.array(
        [
            r
            for episode in toolbox.load_dataset(dataset_index)
            for r in episode["rewards"]
        ]
    )[indices].tolist()
    discriminator = DiscriminatorsWithDirectInput()
    result = getattr(discriminator, algorithm)(
        env_indices=env_indices,
        q_cur_tables=q_cur_tables,
        v_next_tables=v_next_tables,
        all_rewards=all_rewards,
    )
    os.makedirs(
        f"offline_data/multiple_rollouts_cache/selection_results/"
        f"{trial_index}/{grid_index}/{algorithm}/{policy_index}/{dataset_index}",
        exist_ok=True,
    )
    toolbox.save_as_pkl(
        f"offline_data/multiple_rollouts_cache/selection_results/"
        f"{trial_index}/{grid_index}/{algorithm}/{policy_index}/{dataset_index}/result",
        result,
    )
    return result


def write_dicts_to_csv(file_name, dict_list):
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=dict_list[0].keys())
        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)


def observe_multiple_rollouts(num_trials, grid_size, baselines):
    observer = AblationObserver()
    drop_iid_trials(grid_size)
    subset_sizes = [
        int(x) for x in np.linspace(2, parser.args.rollout_num_trajectories, grid_size)
    ]
    j_table, l_table = observer.pre_compute(
        num_states=1024, gamma=parser.args.gamma, horizon=observer.INF
    )
    post_data = []
    multi_args = [
        (algorithm, trial_index, grid_index, policy_index, dataset_index)
        for dataset_index in dataset_indices
        for algorithm in baselines
        for grid_index in range(grid_size)
        for policy_index in policy_indices
        for trial_index in range(num_trials)
    ]
    num_cores = available_cpu_cores(percent=10, limit=24)
    with Pool(processes=num_cores) as pool:
        selection_results = list(
            tqdm(
                pool.imap(selection_with_realizability, multi_args),
                total=len(multi_args),
                desc="Doing selections on multiple rollouts...",
            )
        )
    count = 0
    for dataset_index in dataset_indices:
        for algorithm in baselines:
            j_errors_all = [[] for _ in range(grid_size)]
            for grid_index in range(grid_size):
                for policy_index in policy_indices:
                    for trial_index in range(num_trials):
                        rank, loss = selection_results[count]
                        count += 1
                        if algorithm == "trivial_random":
                            j_error = np.mean(
                                [
                                    math.fabs(
                                        j_table[env_hat][policy_index]
                                        - j_table[dataset_index][policy_index]
                                    )
                                    for env_hat in env_indices
                                ]
                            )
                        else:
                            env_hat = rank[0]
                            j_error = math.fabs(
                                j_table[env_hat][policy_index]
                                - j_table[dataset_index][policy_index]
                            )
                        j_errors_all[grid_index].append(j_error)
            j_errors_mean, j_errors_low, j_errors_high = zip(
                *[
                    stat_from_seq(j_errors, confidence=95, num_resamples=512)
                    for j_errors in j_errors_all
                ]
            )
            data = {
                "ground_id": dataset_index,
                "alg": algorithm,
                "x_axis": subset_sizes,
                "y_mean": j_errors_mean,
                "y_low": j_errors_low,
                "y_high": j_errors_high,
            }
            post_data.append(data)
            plt.plot(subset_sizes, j_errors_mean, label=f"{algorithm}")
            plt.fill_between(subset_sizes, j_errors_low, j_errors_high, alpha=0.2)
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.title(f"Dataset: {dataset_index}")
        plt.xlabel(f"Number of independent rollouts applied in function estimates")
        plt.ylabel("J error without normalization")
        plt.tight_layout()
        os.makedirs(f"offline_data/figures/multiple_rollouts", exist_ok=True)
        plt.savefig(f"offline_data/figures/multiple_rollouts/{dataset_index}.pdf")
        plt.close()
        write_dicts_to_csv("multiple_rollouts.csv", post_data)


def observe_loss(num_trials, grid_size, baselines):
    observer = AblationObserver()
    drop_iid_trials(grid_size)
    subset_sizes = [
        int(x) for x in np.linspace(2, parser.args.rollout_num_trajectories, grid_size)
    ]
    j_table, l_table = observer.pre_compute(
        num_states=1024, gamma=parser.args.gamma, horizon=observer.INF
    )
    post_data = []
    multi_args = [
        (algorithm, trial_index, grid_index, policy_index, dataset_index)
        for dataset_index in dataset_indices
        for algorithm in baselines
        for grid_index in range(grid_size)
        for policy_index in policy_indices
        for trial_index in range(num_trials)
    ]
    num_cores = available_cpu_cores(percent=10, limit=24)
    with Pool(processes=num_cores) as pool:
        selection_results = list(
            tqdm(
                pool.imap(selection_with_realizability, multi_args),
                total=len(multi_args),
                desc="Doing selections on multiple rollouts...",
            )
        )
    count = 0
    for dataset_index in dataset_indices:
        for algorithm in baselines:
            if algorithm == "trivial_random":
                count += grid_size * len(policy_indices) * num_trials
                continue
            loss_all = [[] for _ in range(grid_size)]
            for grid_index in range(grid_size):
                for policy_index in policy_indices:
                    for trial_index in range(num_trials):
                        rank, loss = selection_results[count]
                        count += 1
                        loss_all[grid_index].append(loss)
                # loss_all[grid_index] = np.mean(np.array(loss_all[grid_index]), axis=0, keepdims=False).tolist()
            loss_all = (
                np.array(loss_all).transpose(2, 0, 1).tolist()
            )  # [E, G, N] after permutation
            for env_index, loss in enumerate(loss_all):
                loss_mean, loss_low, loss_high = zip(
                    *[
                        stat_from_seq(iid_loss, confidence=95, num_resamples=512)
                        for iid_loss in loss
                    ]
                )
                post_data.append(
                    {
                        "ground_id": dataset_index,
                        "alg": algorithm,
                        "env": env_index,
                        "x_axis": subset_sizes,
                        "y_mean": loss_mean,
                        "y_low": loss_low,
                        "y_high": loss_high,
                    }
                )
                plt.plot(
                    subset_sizes,
                    loss_mean,
                    label=f"Env {env_index}",
                    marker=markers[env_index],
                )
                plt.fill_between(subset_sizes, loss_low, loss_high, alpha=0.2)
            plt.legend()
            plt.title(f"Dataset: {dataset_index}, Algorithm: {algorithm}")
            os.makedirs(f"offline_data/figures/multiple_rollouts_loss/", exist_ok=True)
            plt.savefig(
                f"offline_data/figures/multiple_rollouts/{dataset_index}_{algorithm}.pdf"
            )
            plt.close()
            write_dicts_to_csv("multiple_rollouts_loss.csv", post_data)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    baselines = [
        "trivial_random",
        "model_free_lstdq_minus",
        "model_free_lstdq_no_minus",
        "model_free_hybrid_lstdq",
        "model_free_std_norm_lstdq_no_minus",
        "model_free_std_norm_lstdq_minus",
        "model_free_avg_bellman_error",
    ]
    observe_loss(num_trials=128, grid_size=16, baselines=baselines)
    observe_multiple_rollouts(num_trials=128, grid_size=16, baselines=baselines)
