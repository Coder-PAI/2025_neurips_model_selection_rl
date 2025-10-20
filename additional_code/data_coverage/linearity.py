import csv
import multiprocessing
import os
import parser

import matplotlib.pyplot as plt
from ablation_observer import AblationObserver
from global_utils import GeneralUtils, device
from scipy.stats import pearsonr


def write_dicts_to_csv(file_name, dict_list):
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["policy", "bellman_errors", "j_errors", "corr"]
        )
        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)


if __name__ == "__main__":
    plt.close()
    multiprocessing.set_start_method("spawn")
    o = AblationObserver()
    t = GeneralUtils()
    j_table, l_table, j_stat_errors, l_stat_errors = o.pre_compute(
        1024, 0.99, o.INF, mean_only=False
    )
    policy_index = parser.args.policy_for_traverse[0]
    policy = t.load_policy("DDPG", policy_index, device)
    dataset_indices = parser.args.ground_truth_for_traverse
    env_indices = range(len(parser.args.hopper_gravities))
    env_class = [t.load_env(index) for index in env_indices]
    t.remove_all_placeholders()
    gamma = 0.99

    j_errors, errors = [], []
    post_data = []

    for policy_index in parser.args.policy_for_traverse:
        for target in dataset_indices:
            for env in env_indices:
                dataset = t.load_dataset(target)
                q_functions = t.load_q_function(env, policy_index, target)
                v_functions = t.load_v_function(env, policy_index, target)
                j_error = j_table[env][policy_index] - j_table[target][policy_index]
                error = 0.0
                for episode_q, episode, episode_v in zip(
                    q_functions, dataset, v_functions
                ):
                    assert (
                        len(episode_q) == len(episode_v) - 1 == len(episode["rewards"])
                    )
                    error += sum(
                        gamma**h * (q - r - gamma * v)
                        for h, (q, r, v) in enumerate(
                            zip(episode_q, episode["rewards"], episode_v[1:])
                        )
                    )
                error /= len(dataset)
                plt.scatter(error, j_error)
                errors.append(error)
                j_errors.append(j_error)
        corr, _ = pearsonr(errors, j_errors)
        data = {
            "policy": policy_index,
            "bellman_errors": errors,
            "j_errors": j_errors,
            "corr": corr,
        }
        post_data.append(data)
        plt.xlabel("Summation of Discounted Bellman Loss")
        plt.ylabel("J error")
        plt.title(f"Pearson Correlation: {corr} (Using behavior data)")
        os.makedirs(f"offline_data/occupancy_curve", exist_ok=True)
        plt.savefig(f"offline_data/occupancy_curve/{policy_index}.png")
        plt.close()
    write_dicts_to_csv("occupancy.csv", post_data)
