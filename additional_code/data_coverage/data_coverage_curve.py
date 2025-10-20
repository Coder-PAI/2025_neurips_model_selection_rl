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
from global_utils import GeneralUtils, stat_from_seq
from selection_algorithms import DiscriminatorsWithDirectInput
from tqdm import tqdm


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


def sample_indices_for_once(b_size, e_size, prop_e_list):
    size = min(b_size, e_size)
    indices = list(range(size))
    indices = np.random.choice(indices, size=size, replace=True).tolist()
    e_sub_samples_size = [int(prop * size) for prop in prop_e_list]
    e_sub_samples = [indices[:sample_size] for sample_size in e_sub_samples_size]
    b_sub_samples = [indices[sample_size:] for sample_size in e_sub_samples_size]
    return b_sub_samples, e_sub_samples


def get_behavior_offline_functions_and_data(policy_index, dataset_index):
    toolbox = GeneralUtils()
    if os.path.exists(
        f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}/all_data_b.pkl"
    ):
        return toolbox.load_from_pkl(
            f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}/all_data_b"
        )
    env_indices = list(range(len(parser.args.hopper_gravities)))
    q_cur_tables = []
    v_next_tables = []
    all_rewards = [
        r for episode in load_data_b(dataset_index) for r in episode["rewards"]
    ]
    timestamps = [
        t
        for episode in load_data_b(dataset_index)
        for t, _ in enumerate(episode["rewards"])
    ]
    for env_index in env_indices:
        q_function = [
            q
            for episode in load_q_b(env_index, policy_index, dataset_index)
            for q in episode
        ]
        v_function = [
            v
            for episode in load_v_b(env_index, policy_index, dataset_index)
            for v in episode[1:]
        ]
        assert (
            len(timestamps) == len(q_function) == len(v_function) == len(all_rewards)
        ), f"{len(q_function), len(v_function), len(all_rewards)}"
        q_cur_tables.append(q_function)
        v_next_tables.append(v_function)
    os.makedirs(
        f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}",
        exist_ok=True,
    )
    toolbox.save_as_pkl(
        f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}/all_data_b",
        [timestamps, q_cur_tables, all_rewards, v_next_tables],
    )
    return [timestamps, q_cur_tables, all_rewards, v_next_tables]


def get_target_offline_functions_and_data(policy_index, dataset_index):
    toolbox = GeneralUtils()
    if os.path.exists(
        f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}/all_data_e.pkl"
    ):
        return toolbox.load_from_pkl(
            f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}/all_data_e"
        )
    env_indices = list(range(len(parser.args.hopper_gravities)))
    q_cur_tables = []
    v_next_tables = []
    all_rewards = [
        r for episode in load_data_e(dataset_index) for r in episode["rewards"]
    ]
    timestamps = [
        t
        for episode in load_data_e(dataset_index)
        for t, _ in enumerate(episode["rewards"])
    ]
    for env_index in env_indices:
        q_function = [
            q
            for episode in load_q_e(env_index, policy_index, dataset_index)
            for q in episode
        ]
        v_function = [
            v
            for episode in load_v_e(env_index, policy_index, dataset_index)
            for v in episode[1:]
        ]
        assert (
            len(timestamps) == len(q_function) == len(v_function) == len(all_rewards)
        ), f"{len(q_function), len(v_function), len(all_rewards)}"
        q_cur_tables.append(q_function)
        v_next_tables.append(v_function)
    os.makedirs(
        f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}",
        exist_ok=True,
    )
    toolbox.save_as_pkl(
        f"offline_data/data_coverage_cache/all_data_cache/{policy_index}/{dataset_index}/all_data_e",
        [timestamps, q_cur_tables, all_rewards, v_next_tables],
    )
    return [timestamps, q_cur_tables, all_rewards, v_next_tables]


def store_offline_indices(num_trials, b_size, e_size, prop_e_list):
    toolbox = GeneralUtils()
    for i in range(num_trials):
        b_sub_samples, e_sub_samples = sample_indices_for_once(
            b_size, e_size, prop_e_list
        )
        os.makedirs(f"offline_data/data_coverage_cache/{i}", exist_ok=True)
        toolbox.save_as_pkl(
            f"offline_data/data_coverage_cache/{i}/sub_sample_indices",
            [b_sub_samples, e_sub_samples],
        )


def load_function(env_index, policy_index, dataset_index, load_q, load_behavior):
    source = "behavior" if load_behavior else "target"
    func_type = "q" if load_q else "v"
    toolbox = GeneralUtils()
    function = toolbox.load_from_pkl(
        f"external_data/data_coverage/{source}/{func_type}_functions/{env_index}/{policy_index}/{dataset_index}/{func_type}_function"
    )
    return function


def load_q_b(env_index, policy_index, dataset_index):
    return load_function(
        env_index, policy_index, dataset_index, load_q=True, load_behavior=True
    )


def load_v_b(env_index, policy_index, dataset_index):
    return load_function(
        env_index, policy_index, dataset_index, load_q=False, load_behavior=True
    )


def load_q_e(env_index, policy_index, dataset_index):
    return load_function(
        env_index, policy_index, dataset_index, load_q=True, load_behavior=False
    )


def load_v_e(env_index, policy_index, dataset_index):
    return load_function(
        env_index, policy_index, dataset_index, load_q=False, load_behavior=False
    )


def load_data_e(dataset_index):
    toolbox = GeneralUtils()
    dataset = toolbox.load_from_pkl(
        f"external_data/data_coverage/target/datasets/{dataset_index}/dataset"
    )
    return dataset


def load_data_b(dataset_index):
    toolbox = GeneralUtils()
    dataset = toolbox.load_from_pkl(
        f"external_data/data_coverage/behavior/datasets/{dataset_index}/dataset"
    )
    return dataset


def subset(data, indices):
    return [data[i] for i in indices]


def merge_data(
    timestamps_b,
    q_tables_b,
    rewards_b,
    v_tables_b,
    timestamps_e,
    q_tables_e,
    rewards_e,
    v_tables_e,
    b_sub_indices,
    e_sub_indices,
):
    for q_table_b, q_table_e, v_table_b, v_table_e in zip(
        q_tables_b, q_tables_e, v_tables_b, v_tables_e
    ):
        assert len(timestamps_b) == len(q_table_b) == len(v_table_b) == len(
            rewards_b
        ) and len(timestamps_e) == len(q_table_e) == len(v_table_e) == len(
            rewards_e
        ), f"{len(q_table_b), len(v_table_b), len(rewards_b), len(q_table_e), len(v_table_e), len(rewards_e)}"
    return (
        subset(timestamps_b, b_sub_indices) + subset(timestamps_e, e_sub_indices),
        [
            subset(q_table_b, b_sub_indices) + subset(q_table_e, e_sub_indices)
            for q_table_b, q_table_e in zip(q_tables_b, q_tables_e)
        ],
        subset(rewards_b, b_sub_indices) + subset(rewards_e, e_sub_indices),
        [
            subset(v_table_b, b_sub_indices) + subset(v_table_e, e_sub_indices)
            for v_table_b, v_table_e in zip(v_tables_b, v_tables_e)
        ],
    )


def merge_selection(args):
    algorithm, trial_index, prop_e_index, policy_index, dataset_index = args
    toolbox = GeneralUtils()
    b_sub_indices, e_sub_indices = toolbox.load_from_pkl(
        f"offline_data/data_coverage_cache/{trial_index}/sub_sample_indices"
    )
    b_sub_indices = b_sub_indices[prop_e_index]
    e_sub_indices = e_sub_indices[prop_e_index]
    env_indices = list(range(len(parser.args.hopper_gravities)))
    (
        timestamps_b,
        q_cur_tables_b,
        all_rewards_b,
        v_next_tables_b,
    ) = get_behavior_offline_functions_and_data(policy_index, dataset_index)
    (
        timestamps_e,
        q_cur_tables_e,
        all_rewards_e,
        v_next_tables_e,
    ) = get_target_offline_functions_and_data(policy_index, dataset_index)
    timestamps, q_cur_tables, all_rewards, v_next_tables = merge_data(
        timestamps_b,
        q_cur_tables_b,
        all_rewards_b,
        v_next_tables_b,
        timestamps_e,
        q_cur_tables_e,
        all_rewards_e,
        v_next_tables_e,
        b_sub_indices,
        e_sub_indices,
    )

    discriminator = DiscriminatorsWithDirectInput()
    try:
        result = getattr(discriminator, algorithm)(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )
    except:
        result = getattr(discriminator, algorithm)(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
            timestamps=timestamps,
        )
    return result


def write_dicts_to_csv(file_name, dict_list):
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=dict_list[0].keys())
        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)


def observe_data_coverage(baselines, num_props, num_trials, num_states, gamma):
    def check(args_1, args_2):
        return len(args_1) == len(args_2) and all(
            arg_1 == arg_2 for arg_1, arg_2 in zip(args_1, args_2)
        )

    observer = AblationObserver()
    toolbox = GeneralUtils()
    os.makedirs(f"offline_data/data_coverage_cache", exist_ok=True)
    j_table, l_table = observer.pre_compute(
        num_states, gamma, observer.INF, enable_pool=True
    )
    dataset_indices = parser.args.ground_truth_for_traverse
    env_indices = list(range(len(parser.args.hopper_gravities)))

    props = np.linspace(0, 1, num_props)
    b_size = e_size = parser.args.sampler_size

    store_offline_indices(num_trials, b_size, e_size, props)
    post_data = []

    multi_args = []
    assert (
        len(parser.args.policy_for_traverse) == 1
    ), "Data coverage experiments must have a single policy only!"
    policy_index = parser.args.policy_for_traverse[0]
    for dataset_index in dataset_indices:
        get_behavior_offline_functions_and_data(policy_index, dataset_index)
        get_target_offline_functions_and_data(policy_index, dataset_index)
    for dataset_index in dataset_indices:
        for algorithm in baselines:
            for trial_index in range(num_trials):
                for prop_index, _ in enumerate(props):
                    multi_args.append(
                        [
                            algorithm,
                            trial_index,
                            prop_index,
                            policy_index,
                            dataset_index,
                        ]
                    )
    num_cores = available_cpu_cores(percent=10, limit=24)
    try:
        selection_results = toolbox.load_from_pkl(
            f"offline_data/data_coverage_cache/results"
        )
    except FileNotFoundError:
        with Pool(processes=num_cores) as pool:
            selection_results = list(
                tqdm(
                    pool.imap(merge_selection, multi_args),
                    total=len(multi_args),
                    desc="Observing data coverage...",
                )
            )
            selection_results = [
                [arg, result] for arg, result in zip(multi_args, selection_results)
            ]
        toolbox.save_as_pkl(
            f"offline_data/data_coverage_cache/results", selection_results
        )
    count = 0
    for dataset_index in dataset_indices:
        for algorithm in baselines:
            cache = [[] for _ in props]
            for trial_index in range(num_trials):
                for prop_index, _ in enumerate(props):
                    arg, [ranking, _] = selection_results[count]
                    check([dataset_index, algorithm, trial_index, prop_index], arg)
                    count += 1
                    env_hat = ranking[0]
                    j_error = math.fabs(
                        j_table[env_hat][policy_index]
                        - j_table[dataset_index][policy_index]
                    )
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
                    cache[prop_index].append(j_error)
            j_errors_mean, j_errors_low, j_errors_high = zip(
                *[
                    stat_from_seq(estimates, confidence=95, num_resamples=512)
                    for estimates in cache
                ]
            )
            plt.plot(props, j_errors_mean, label=f"{algorithm}")
            plt.fill_between(props, j_errors_low, j_errors_high, alpha=0.2)
            post_data.append(
                {
                    "ground_id": dataset_index,
                    "alg": algorithm,
                    "x_axis": props,
                    "y_mean": j_errors_mean,
                    "y_low": j_errors_low,
                    "y_high": j_errors_high,
                }
            )
        plt.legend()
        plt.rcParams["font.size"] = 8
        plt.xlabel("Proportion of the dataset sampled by target policy")
        plt.ylabel("Un-normalized J Error")
        plt.title(
            f"Investigation of Data Coverage on the Selection Problem of\n"
            f"target_policy={policy_index}, dataset_index={dataset_index}"
        )
        os.makedirs(f"offline_data/figures/data_coverage/", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/data_coverage/data_coverage_{dataset_index}.pdf"
        )
        plt.close()
        write_dicts_to_csv(f"data_coverage.csv", post_data)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    baselines = [
        "trivial_random",
        "model_free_td_square",
        "model_free_avg_bellman_error",
        "model_free_hybrid_lstdq",
        "model_free_bvft",
    ]
    num_props = 8
    num_trials = 256
    num_states = 1024
    gamma = parser.args.gamma
    observe_data_coverage(baselines, num_props, num_trials, num_states, gamma)
