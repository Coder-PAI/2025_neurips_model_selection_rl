import copy
import math
import os
import parser
import random
import time
from multiprocessing import Pool

import gym
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from tqdm import tqdm

from global_utils import GeneralUtils, device, stat_from_seq
from model_based_algorithms import model_based_alg
from model_free_algorithms import model_free_alg
from selection_algorithms import SelectionBaselines

plt.rcParams["font.size"] = 8


def subset(data, indices):
    return [data[i] for i in indices]


def rollout_single_trajectory(env, policy, gamma, horizon):
    j_estimation = 0.0
    l_estimation = 0.0
    obs, _ = env.reset()
    for t in range(horizon):
        l_estimation += 1.0
        action = policy.predict(obs)[0]
        obs, reward, done, _, _ = env.step(action)
        j_estimation += reward * gamma**t
        if done:
            break
    env.close()
    return j_estimation, l_estimation


def rollout_single_trajectory_as_single_task(args):
    toolbox, env_index, policy_index, gamma, horizon = args
    env = toolbox.load_env(env_index)
    policy = toolbox.load_policy("DDPG", policy_index, device)
    return rollout_single_trajectory(env, policy, gamma, horizon)


def selection_as_single_task(args):
    if len(args) == 5:
        algorithm, policy_index, candidates, target, resample_size = args
        resample_indices = None
    elif len(args) == 6:
        (
            algorithm,
            policy_index,
            candidates,
            target,
            resample_size,
            resample_indices,
        ) = args
    else:
        raise Exception(f"Unexpected number of arguments: {len(args)}!")
    selector = SelectionBaselines()
    return getattr(selector, algorithm)(
        policy_index=policy_index,
        env_indices=candidates,
        dataset_index=target,
        sample_size=resample_size,
        specify_indices=resample_indices,
    )


def incremental_selection_as_single_task(args):
    (
        trial_index,
        algorithm,
        policy_index,
        candidates,
        target,
        resample_size,
        sample_range,
    ) = args
    selector = SelectionBaselines(method="incremental", sample_range=sample_range)
    result = [
        rank[0]
        for rank, loss in getattr(selector, algorithm)(
            policy_index=policy_index,
            env_indices=candidates,
            dataset_index=target,
            sample_size=resample_size,
            specify_indices=None,
        )
    ]
    return result


def sample_curve(args):
    selector = SelectionBaselines()
    (
        algorithm,
        policy_index,
        dataset_index,
        sample_range,
        iid_resample_times,
        j_table,
    ) = args
    curve_values = []
    env_indices = list(range(len(parser.args.hopper_gravities)))
    for sample_size in sample_range:
        # print(policy_index, dataset_index, sample_size)
        iid_j_error = []
        for _ in range(iid_resample_times):
            ranked_env_indices, loss_over_all_envs = getattr(selector, algorithm)(
                policy_index=policy_index,
                env_indices=env_indices,
                dataset_index=dataset_index,
                sample_size=sample_size,
            )
            env_hat = ranked_env_indices[0]
            iid_j_error.append(
                math.fabs(
                    j_table[dataset_index][policy_index]
                    - j_table[env_hat][policy_index]
                )
            )
        # observe the j error
        curve_values.append(np.mean(iid_j_error))
    return curve_values


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


def gather_initial_states(toolbox, num_states):
    if os.path.exists(f"offline_data/initial_states/data_{num_states}.pkl"):
        # self.toolbox.render_text(f"Initial states sized {num_states} has already been collected. "
        # f"Loading states from cached data...", "RED")
        return toolbox.load_from_pkl(f"offline_data/initial_states/data_{num_states}")

    initial_states = []
    env = gym.make(
        "Hopper-v4",
    )
    for _ in range(num_states):
        obs, _ = env.reset()
        q_pos = copy.deepcopy(env.data.qpos[:])
        q_vel = copy.deepcopy(env.data.qvel[:])
        initial_states.append([obs, q_pos, q_vel])
    env.close()
    os.makedirs(f"offline_data/initial_states/", exist_ok=True)
    toolbox.save_as_pkl(
        f"offline_data/initial_states/data_{num_states}", initial_states
    )
    return initial_states


def rollout_with_states_specified(
    toolbox,
    env,
    policy,
    num_states,
    gamma,
    horizon,
    bar=False,
    mean_only=True,
    trim_prop=0.1,
):
    initial_states = gather_initial_states(toolbox, num_states)
    j_estimations = []
    l_estimations = []
    for state, qpos, qvel in initial_states if not bar else tqdm(initial_states):
        t = 0
        env.reset()
        env.set_state(qpos, qvel)
        obs = state
        j_estimations.append(0.0)
        l_estimations.append(0.0)
        for _ in range(horizon):
            l_estimations[-1] += 1.0
            action = policy.predict(obs)[0]
            obs, reward, done, _, _ = env.step(action)
            j_estimations[-1] += reward * gamma**t
            t += 1
            if done:
                break
    j_mean, j_low, j_high = stat_from_seq(
        j_estimations, confidence=95, num_resamples=1024, trim_prop=trim_prop
    )
    l_mean, l_low, l_high = stat_from_seq(
        l_estimations, confidence=95, num_resamples=1024, trim_prop=trim_prop
    )
    env.close()
    if mean_only:
        return j_mean, l_mean
    else:
        return j_mean, l_mean, (j_low, j_high), (l_low, l_high)


def j_estimation_as_single_task(args):
    toolbox, env_index, policy_index, num_states, gamma, horizon = args
    policy = toolbox.load_policy("DDPG", policy_index, device)
    env = toolbox.load_env(env_index)
    j_table_path = f"offline_data/j_tables/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}.pkl"
    l_table_path = f"offline_data/l_tables/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}.pkl"
    j_stat_errors_path = f"offline_data/j_stat_errors/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}.pkl"
    l_stat_errors_path = f"offline_data/l_stat_errors/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}.pkl"
    if all(
        [
            os.path.exists(path)
            for path in [
                j_table_path,
                l_table_path,
                j_stat_errors_path,
                l_stat_errors_path,
            ]
        ]
    ):
        return list(
            toolbox.load_from_pkl(path.replace(".pkl", ""))
            for path in [
                j_table_path,
                l_table_path,
                j_stat_errors_path,
                l_stat_errors_path,
            ]
        )
    else:
        os.makedirs(f"offline_data/j_tables/{env_index}/{policy_index}", exist_ok=True)
        os.makedirs(f"offline_data/l_tables/{env_index}/{policy_index}", exist_ok=True)
        os.makedirs(
            f"offline_data/j_stat_errors/{env_index}/{policy_index}", exist_ok=True
        )
        os.makedirs(
            f"offline_data/l_stat_errors/{env_index}/{policy_index}", exist_ok=True
        )
        (
            j_estimation,
            l_estimation,
            j_conf_interval,
            l_conf_interval,
        ) = rollout_with_states_specified(
            toolbox, env, policy, num_states, gamma, horizon, bar=True, mean_only=False
        )
        toolbox.save_as_pkl(
            f"offline_data/j_tables/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}",
            j_estimation,
        )
        toolbox.save_as_pkl(
            f"offline_data/l_tables/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}",
            l_estimation,
        )
        toolbox.save_as_pkl(
            f"offline_data/j_stat_errors/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}",
            j_conf_interval,
        )
        toolbox.save_as_pkl(
            f"offline_data/l_stat_errors/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}",
            l_conf_interval,
        )
        return j_estimation, l_estimation, j_conf_interval, l_conf_interval


class AblationObserver:
    INF = 99999999

    def __init__(self):
        self.toolbox = GeneralUtils()
        self.policy_indices = parser.args.policy_for_traverse
        self.dataset_indices = parser.args.ground_truth_for_traverse
        (
            self.env_class,
            self.policy_class,
            self.datasets,
        ) = self.toolbox.load_env_policy_dataset()
        self.env_indices = list(range(len(self.env_class)))
        self.env_encoder = {}
        self.rollout_with_states_specified = (
            lambda env, policy, num_states, gamma, horizon, bar=False: (
                rollout_with_states_specified(
                    self.toolbox, env, policy, num_states, gamma, horizon, bar=False
                )
            )
        )
        self.gather_initial_states = lambda num_states: gather_initial_states(
            self.toolbox, num_states
        )

    def pre_compute(self, num_states, gamma, horizon, enable_pool=True, mean_only=True):
        j_table = {i: {j: 0.0 for j in self.policy_indices} for i in self.env_indices}
        l_table = {i: {j: 0.0 for j in self.policy_indices} for i in self.env_indices}
        j_stat_errors = {
            i: {j: 0.0 for j in self.policy_indices} for i in self.env_indices
        }
        l_stat_errors = {
            i: {j: 0.0 for j in self.policy_indices} for i in self.env_indices
        }
        for env_index in self.env_indices:
            env = self.toolbox.load_env(env_index)
            gravity = env.unwrapped.model.opt.gravity[2]
            noise = env.force_scaler
            self.env_encoder[(gravity, noise)] = env_index
        if enable_pool:
            multi_args = [
                (self.toolbox, env_index, policy_index, num_states, gamma, horizon)
                for env_index in tqdm(self.env_indices)
                for policy_index in self.policy_indices
            ]
            num_cores = available_cpu_cores(percent=10, limit=20)
            with Pool(processes=num_cores) as pool:
                pool.map(j_estimation_as_single_task, multi_args)
        else:
            for env_index in tqdm(self.env_indices):
                for policy_index in self.policy_indices:
                    arg = (
                        self.toolbox,
                        env_index,
                        policy_index,
                        num_states,
                        gamma,
                        horizon,
                    )
                    j_estimation_as_single_task(arg)

        for env_index in self.env_indices:
            for policy_index in self.policy_indices:
                j_estimation, l_estimation, j_conf_interval, l_conf_interval = (
                    self.toolbox.load_from_pkl(
                        f"offline_data/j_tables/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}"
                    ),
                    self.toolbox.load_from_pkl(
                        f"offline_data/l_tables/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}"
                    ),
                    self.toolbox.load_from_pkl(
                        f"offline_data/j_stat_errors/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}"
                    ),
                    self.toolbox.load_from_pkl(
                        f"offline_data/l_stat_errors/{env_index}/{policy_index}/{num_states}_{gamma}_{horizon}"
                    ),
                )
                j_table[env_index][policy_index] = j_estimation
                l_table[env_index][policy_index] = l_estimation
                j_stat_errors[env_index][policy_index] = j_conf_interval
                l_stat_errors[env_index][policy_index] = l_conf_interval
        if mean_only:
            return j_table, l_table
        else:
            return j_table, l_table, j_stat_errors, l_stat_errors

    def check_j_convergence(
        self,
        env,
        policy,
        num_iid_samples,
        gamma,
        horizon,
        confidence,
        num_resamples,
        trim_prop,
    ):
        samples_iid = []
        j_mean_list = []
        j_low_list = []
        j_high_list = []
        for _ in range(num_iid_samples):
            j, _ = rollout_single_trajectory(env, policy, gamma=gamma, horizon=horizon)
            samples_iid.append(j)
            j_mean, j_low, j_high = stat_from_seq(
                samples_iid, confidence, num_resamples, trim_prop=trim_prop
            )
            j_mean_list.append(j_mean)
            j_low_list.append(j_low)
            j_high_list.append(j_high)
        return j_mean_list, j_low_list, j_high_list

    def plot_j_convergence(
        self,
        confidence,
        num_resamples,
        num_iid_samples,
        gamma,
        horizon,
        env_limit,
        start_point,
        trim_prop=0.1,
    ):
        if os.path.exists(
            f"offline_data/figures/j_convergence/figure_{num_iid_samples}_{gamma}_{horizon}.pdf"
        ):
            return
        fig_size = len(self.policy_indices)
        cols = math.ceil(math.sqrt(fig_size))
        rows = math.ceil(fig_size / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()
        sample_times = list(range(1, num_iid_samples + 1))
        env_gap = len(self.env_indices) // env_limit
        env_indices = (
            [self.env_indices[i * env_gap] for i in range(env_limit)]
            if env_gap > 0
            else range(len(parser.args.hopper_gravities))
        )
        bar = tqdm(
            total=len(self.policy_indices) * env_limit, desc=f"Observing convergence..."
        )
        for i, policy_index in enumerate(self.policy_indices):
            policy = self.toolbox.load_policy("DDPG", policy_index, device)
            for env_index in env_indices:
                bar.update(1)
                env = self.toolbox.load_env(env_index)
                j_mean_list, j_low_list, j_high_list = self.check_j_convergence(
                    env,
                    policy,
                    num_iid_samples,
                    gamma,
                    horizon,
                    confidence,
                    num_resamples,
                    trim_prop,
                )
                axes[i].plot(
                    sample_times[start_point:],
                    j_mean_list[start_point:],
                    label=self.get_main_param(env),
                )
                axes[i].fill_between(
                    sample_times[start_point:],
                    j_low_list[start_point:],
                    j_high_list[start_point:],
                    alpha=0.2,
                )
                axes[i].legend()
                axes[i].set_title(f"J Convergence Under Policy {policy_index}")
                axes[i].set_xlabel("Sample Times")
                axes[i].set_ylabel("J Estimation")
        for j in range(fig_size, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(
            f"J Convergence (with gamma={gamma}, horizon={horizon}, \n"
            f"confidence level={confidence}, number of resamples={num_resamples})"
        )
        plt.tight_layout()
        os.makedirs(f"offline_data/figures/j_convergence", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/j_convergence/figure_{num_iid_samples}_{gamma}_{horizon}.pdf"
        )
        plt.close()

    @staticmethod
    def get_main_param(env):
        gravity = env.unwrapped.model.opt.gravity[2]
        noise = env.force_scaler
        return f"g={gravity:.3f}, ε={noise:.3f}"

    @staticmethod
    def group_by(envs, mode):
        assert mode in ["gravity", "noise"]

        def get_order(env):
            return env.unwrapped.model.opt.gravity[2], env.force_scaler

        bucket = {}
        for env in envs:
            gravity, noise = get_order(env)
            if mode == "gravity":
                if noise not in bucket:
                    bucket[noise] = []
                bucket[noise].append((env, gravity))
            elif mode == "noise":
                if gravity not in bucket:
                    bucket[gravity] = []
                bucket[gravity].append((env, noise))
        for label in bucket:
            bucket[label] = sorted(bucket[label], key=lambda x: x[1], reverse=False)
        return bucket

    def plot_j_with_param_differ(self, num_states, gamma, horizon, mode):
        if os.path.exists(
            f"offline_data/figures/j_curve/figure_{num_states}_{gamma}_{horizon}.pdf"
        ):
            return
        j_table, l_table, j_stat_errors, l_stat_errors = self.pre_compute(
            num_states, gamma, horizon, enable_pool=True, mean_only=False
        )
        envs = [self.toolbox.load_env(env_index) for env_index in self.env_indices]
        bucket = self.group_by(envs, mode)
        bar = tqdm(
            total=len(self.policy_indices) * len(self.env_indices),
            desc="Plotting J Curve: ",
        )
        for i, policy_index in enumerate(self.policy_indices):
            for label_1 in bucket:
                x_labels = []
                j_list = []
                j_conf_low = []
                j_conf_high = []
                for env, label_2 in bucket[label_1]:
                    param = (
                        (label_2, label_1) if mode == "gravity" else (label_1, label_2)
                    )
                    bar.update(1)
                    x_labels.append(label_2)
                    env_index = self.env_encoder[param]
                    j_estimation = j_table[env_index][policy_index]
                    j_list.append(j_estimation)
                    j_low, j_high = j_stat_errors[env_index][policy_index]
                    j_conf_low.append(j_low)
                    j_conf_high.append(j_high)
                plt.plot(
                    x_labels,
                    j_list,
                    label=f"Policy {policy_index}, "
                    + (f"ε={label_1:.3f}" if mode == "gravity" else f"g={label_1:.3f}"),
                )
                plt.fill_between(x_labels, j_conf_low, j_conf_high, alpha=0.2)
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.xlabel("Gravity" if mode == "gravity" else "Noise")
        plt.ylabel("J Estimation")
        # plt.title(f"J Curve Under Policy {policy_index}")
        plt.tight_layout()
        os.makedirs(f"offline_data/figures/j_curve", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/j_curve/figure_{num_states}_{gamma}_{horizon}.pdf"
        )
        plt.close()

    @staticmethod
    def get_param(gravity, noise):
        # gravity has to be negative!
        return {
            "gravity": [0, 0, gravity],
            "force_mean": 0,
            "force_scaler": noise,
        }

    def observe_param_gap(
        self, num_states, gamma, horizon, resample_size, num_resamples, confidence
    ):
        if os.path.exists(
            f"offline_data/figures/ablation_param_gaps/figure_{num_states}_{gamma}_{horizon}.pdf"
        ):
            return
        if len(set(parser.args.hopper_gravities)) == 1:
            mode = "noise"
        elif len(set(parser.args.hopper_noise_scaler_list)) == 1:
            mode = "gravity"
        else:
            raise AssertionError("Only 1D param grids are allowed for ablation study!")

        def is_arithmetic(param_list):
            return (
                len(param_list) <= 1
                or len(
                    set(
                        f"{(y - x):.6f}"
                        for x, y in zip(param_list[:-1], param_list[1:])
                    )
                )
                == 1
            )

        assert is_arithmetic(parser.args.hopper_gravities)
        assert is_arithmetic(parser.args.hopper_noise_scaler_list)

        def has_center(param_list):
            return len(param_list) % 2 == 1

        # assert has_center(parser.args.hopper_gravities)
        # assert has_center(parser.args.hopper_noise_scaler_list)

        selector = SelectionBaselines()
        baselines = selector.baselines
        j_table, l_table = self.pre_compute(num_states, gamma, horizon)
        if mode == "gravity":
            center_loc = len(parser.args.hopper_gravities) // 2
            gaps = sorted(
                [
                    math.fabs(
                        parser.args.hopper_gravities[center_loc]
                        - parser.args.hopper_gravities[i]
                    )
                    for i in range(
                        1 if not has_center(parser.args.hopper_gravities) else 0,
                        center_loc,
                    )
                ]
            )
        else:
            center_loc = len(parser.args.hopper_noise_scaler_list) // 2
            gaps = sorted(
                [
                    math.fabs(
                        parser.args.hopper_noise_scaler_list[center_loc]
                        - parser.args.hopper_noise_scaler_list[i]
                    )
                    for i in range(
                        1
                        if not has_center(parser.args.hopper_noise_scaler_list)
                        else 0,
                        center_loc,
                    )
                ]
            )

        multi_args = [
            (
                algorithm,
                policy_index,
                [center_loc - (index + 1), center_loc, center_loc + (index + 1)],
                center_loc,
                resample_size,
            )
            for algorithm in baselines
            for index, gap in enumerate(gaps)
            for policy_index in self.policy_indices
            for _ in range(num_resamples)
        ]

        num_cores = available_cpu_cores(percent=10, limit=24)
        with Pool(processes=num_cores) as pool:
            selection_results = list(
                tqdm(
                    pool.imap(selection_as_single_task, multi_args),
                    total=len(multi_args),
                    desc="Observing J-Error(Gap of Params)...",
                )
            )
        count = 0
        for algorithm in baselines:
            j_errors_specified_alg = []
            j_errors_conf_low = []
            j_errors_conf_high = []
            for index, gap in enumerate(gaps):
                resample_j_errors = []
                for policy_index in self.policy_indices:
                    target = center_loc
                    for _ in range(num_resamples):
                        ranked_env_indices, loss = selection_results[count]
                        count += 1
                        env_hat = ranked_env_indices[0]
                        j_error = math.fabs(
                            j_table[target][policy_index]
                            - j_table[env_hat][policy_index]
                        )
                        resample_j_errors.append(j_error)
                j_err_mean, j_err_low, j_err_high = stat_from_seq(
                    resample_j_errors, confidence, num_resamples
                )
                j_errors_specified_alg.append(j_err_mean)
                j_errors_conf_low.append(j_err_low)
                j_errors_conf_high.append(j_err_high)
            # j_error_specified_gap.append(np.mean(j_errors_specified_gap))
            plt.plot(gaps, j_errors_specified_alg, label=algorithm)
            plt.fill_between(gaps, j_errors_conf_low, j_errors_conf_high, alpha=0.2)
        plt.legend()
        plt.title(
            f"J error under different resolutions "
            + (
                f"(ε={parser.args.hopper_noise_scaler_list[0]})"
                if mode == "gravity"
                else f"(g={parser.args.hopper_gravities[0]})"
            )
        )
        plt.xlabel(f"Gap of " + mode)
        plt.ylabel("Un-normalized J Error")
        os.makedirs(f"offline_data/figures/ablation_param_gaps", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/ablation_param_gaps/figure_{num_states}_{gamma}_{horizon}.pdf"
        )
        plt.close()

    @staticmethod
    def forward(env, q_pos, q_vel, action):
        env.reset()
        env.set_state(q_pos, q_vel)
        obs, reward, done, _, _ = env.step(action)
        return obs, reward, done

    def transition_noisy_measure(self, policy, states_pos_vel, gravity, noise_scaler):
        param_noisy = self.get_param(gravity, noise_scaler)
        param_zero_noisy = self.get_param(gravity, 0.0)
        with_noise = self.toolbox.hopper_create_with_params(param_noisy)
        zero_noise = self.toolbox.hopper_create_with_params(param_zero_noisy)
        with_noise_transitions = []
        zero_noise_transitions = []
        for state, q_pos, q_vel in states_pos_vel:
            action = policy.predict(state)[0]
            obs_with_noise, _, _ = self.forward(with_noise, q_pos, q_vel, action)
            obs_without_noise, _, _ = self.forward(zero_noise, q_pos, q_vel, action)
            with_noise_transitions.append(obs_with_noise)
            zero_noise_transitions.append(obs_without_noise)
        biases = [
            obs_with_noise - obs_without_noise
            for obs_with_noise, obs_without_noise in zip(
                with_noise_transitions, zero_noise_transitions
            )
        ]
        return np.mean([np.linalg.norm(bias) for bias in biases])

    def transition_step_measure(self, policy, states_pos_vel, gravity):
        param_zero_noisy = self.get_param(gravity, 0.0)
        zero_noise = self.toolbox.hopper_create_with_params(param_zero_noisy)
        step_transitions = []
        for state, q_pos, q_vel in states_pos_vel:
            action = policy.predict(state)[0]
            obs_without_noise, _, _ = self.forward(zero_noise, q_pos, q_vel, action)
            step_transitions.append(obs_without_noise)
        biases = [
            obs_without_noise - state
            for obs_without_noise, (state, _, _) in zip(
                step_transitions, states_pos_vel
            )
        ]
        return np.mean([np.linalg.norm(bias) for bias in biases])

    def observe_noise(self, num_states, gravity, noise_list):
        if os.path.exists(
            f"offline_data/figures/ablation_noise/figure_{num_states}_{gravity}.pdf"
        ):
            return
        _, _, datasets = self.toolbox.load_env_policy_dataset()
        states, q_pos_list, q_vel_list, _, _, _, _, _ = zip(
            *self.toolbox.all_data_from_datasets(datasets)
        )
        states_pos_vel = list(zip(states, q_pos_list, q_vel_list))
        states_pos_vel = random.sample(states_pos_vel, num_states)
        for policy_index in self.policy_indices:
            policy = self.toolbox.load_policy("DDPG", policy_index, device)
            noisy_measures = [
                self.transition_noisy_measure(
                    policy, states_pos_vel, gravity, noise_scaler
                )
                for noise_scaler in tqdm(noise_list)
            ]
            step_measure = self.transition_step_measure(policy, states_pos_vel, gravity)
            ratio_with_noise_varies = [
                noisy_measure / step_measure for noisy_measure in noisy_measures
            ]
            # print(ratio_with_noise_varies)
            plt.plot(
                noise_list, ratio_with_noise_varies, label=f"Policy {policy_index}"
            )
        plt.title(f"The effect of noise (with g={gravity})")
        plt.xlabel("Noise level")
        plt.ylabel("Ratio")
        plt.legend()
        os.makedirs(f"offline_data/figures/ablation_noise", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/ablation_noise/figure_{num_states}_{gravity}.pdf"
        )
        plt.close()

    def plot_j_distribution(
        self, policy_indices, gravity, noise_scaler, num_iid_samples, gamma, horizon
    ):
        if os.path.exists(
            f"offline_data/figures/j_distribution/j_distribution_{gamma}_{horizon}.pdf"
        ):
            return
        param = self.get_param(gravity, noise_scaler)
        env = self.toolbox.hopper_create_with_params(param)
        for policy_index in policy_indices:
            policy = self.toolbox.load_policy("DDPG", policy_index, device)
            j_samples = [
                rollout_single_trajectory(env, policy, gamma, horizon)[0]
                for _ in tqdm(range(num_iid_samples))
            ]
            sns.kdeplot(j_samples, fill=False, label=f"Policy {policy_index}")
        plt.legend()
        plt.xlabel("Estimated Return")
        plt.ylabel("Probability Density Function")
        plt.title(
            f"PDF for the estimated return"
            f"\n(with g={gravity}, ε={noise_scaler}, gamma={gamma}, horizon={horizon})"
        )
        os.makedirs(f"offline_data/figures/j_distribution", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/j_distribution/j_distribution_{gamma}_{horizon}.pdf"
        )
        plt.close()

    def early_observe(self):
        if len(set(parser.args.hopper_gravities)) == 1:
            mode = "noise"
            (g1, n1), (g2, n2) = [
                (
                    parser.args.hopper_gravities[0],
                    min(parser.args.hopper_noise_scaler_list),
                ),
                (
                    parser.args.hopper_gravities[0],
                    max(parser.args.hopper_noise_scaler_list),
                ),
            ]
        elif len(set(parser.args.hopper_noise_scaler_list)) == 1:
            mode = "gravity"
            (g1, n1), (g2, n2) = [
                (
                    min(parser.args.hopper_gravities),
                    parser.args.hopper_noise_scaler_list[0],
                ),
                (
                    max(parser.args.hopper_gravities),
                    parser.args.hopper_noise_scaler_list[0],
                ),
            ]
        else:
            raise AssertionError(f"Only 1D sequence is accepted for ablation study!")
        # has to be called after all the policies and datasets are fully prepared
        self.plot_j_with_param_differ(
            num_states=1024, gamma=parser.args.gamma, horizon=self.INF, mode=mode
        )
        self.observe_termination(
            num_states=1024, gamma=parser.args.gamma, horizon=self.INF
        )
        self.plot_j_convergence(
            confidence=95,
            num_resamples=1024,
            num_iid_samples=1024,
            gamma=parser.args.gamma,
            horizon=self.INF,
            env_limit=4,
            start_point=8,
        )
        self.plot_j_distribution(
            policy_indices=self.policy_indices,
            gravity=g1,
            noise_scaler=n1,
            num_iid_samples=1024,
            gamma=parser.args.gamma,
            horizon=parser.args.rollout_max_horizon,
        )
        self.plot_j_distribution(
            policy_indices=self.policy_indices,
            gravity=g2,
            noise_scaler=n2,
            num_iid_samples=1024,
            gamma=parser.args.gamma,
            horizon=parser.args.rollout_max_horizon,
        )
        self.observe_noise(
            num_states=1024, gravity=-30, noise_list=list(np.linspace(0, 256, 32))
        )

    def observe_termination(self, num_states, gamma, horizon):
        policy_indices = parser.args.policy_for_traverse
        env_indices = list(range(len(parser.args.hopper_gravities)))
        j_table, l_table, j_stat_errors, l_stat_errors = self.pre_compute(
            num_states, gamma, horizon, enable_pool=True, mean_only=False
        )
        for policy_index in tqdm(policy_indices, desc="Observing terminations..."):
            ave_horizons = [
                l_table[env_index][policy_index] for env_index in env_indices
            ]
            horizon_low = [
                l_stat_errors[env_index][policy_index][0] for env_index in env_indices
            ]
            horizon_high = [
                l_stat_errors[env_index][policy_index][1] for env_index in env_indices
            ]
            plt.plot(env_indices, ave_horizons, label=f"Policy {policy_index}")
            plt.fill_between(env_indices, horizon_low, horizon_high, alpha=0.2)
        plt.legend()
        plt.title(
            f"Average horizon of termination in fluctuation with the environments"
        )
        plt.xlabel(f"Indices of Environments")
        plt.ylabel(f"Average Horizon of Termination")
        os.makedirs(f"offline_data/figures/termination", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/termination/termination_{num_states}_{gamma}_{horizon}.pdf"
        )
        plt.close()

    def observe_bvft(self):
        if os.path.exists(
            f"offline_data/figures/bvft_resolutions/bvft_resolutions.pdf"
        ):
            return
        self.toolbox.delete_bvft_cache()
        candidates = list(range(len(parser.args.hopper_gravities)))
        resolutions = list(np.linspace(0.1, 32, 32))
        multi_args = []
        for dataset_index in parser.args.ground_truth_for_traverse:
            for policy_index in parser.args.policy_for_traverse:
                for resolution in resolutions:
                    multi_args.append(
                        (
                            f"model_free_bvft_{resolution}",
                            policy_index,
                            candidates,
                            dataset_index,
                            None,
                            None,
                        )
                    )
        num_cores = available_cpu_cores(percent=10, limit=28)
        with Pool(processes=num_cores) as pool:
            selection_results = list(
                tqdm(
                    pool.imap(selection_as_single_task, multi_args),
                    total=len(multi_args),
                    desc="Observing BVFT resolutions...",
                )
            )
        pointer = 0
        post_data = []
        for dataset_index in parser.args.ground_truth_for_traverse:
            opt_loss_list = [[] for _ in resolutions]
            for _ in parser.args.policy_for_traverse:
                for i, _ in enumerate(resolutions):
                    _, loss_over_all_envs = selection_results[pointer]
                    pointer += 1
                    opt_loss = loss_over_all_envs[dataset_index]
                    opt_loss_list[i].append(opt_loss)
            loss_mean, loss_low, loss_high = zip(
                *[
                    stat_from_seq(loss, confidence=95, num_resamples=512)
                    for loss in opt_loss_list
                ]
            )
            plt.plot(resolutions, loss_mean, label=f"Ground Model {dataset_index}")
            plt.fill_between(resolutions, loss_low, loss_high, alpha=0.2)
        plt.title(f"BVFT losses using different resolutions")
        plt.legend()
        plt.xlabel(f"Resolutions")
        plt.ylabel(f"BVFT Loss")
        os.makedirs(f"offline_data/figures/bvft_resolutions", exist_ok=True)
        plt.savefig(f"offline_data/figures/bvft_resolutions/bvft_resolutions.pdf")
        plt.close()

    def observe_sample_efficiency(
        self, algorithms, path, num_init_states, iid_resample_times, num_samples
    ):
        env_indices = parser.args.ground_truth_for_traverse
        if all(
            [
                os.path.exists(os.path.join(path, f"eff_{dataset_index}.pdf"))
                for dataset_index in env_indices
            ]
        ):
            return
        policy_indices = parser.args.policy_for_traverse
        candidates = list(range(len(parser.args.hopper_gravities)))
        multi_args = []
        j_table, _ = self.pre_compute(
            num_states=num_init_states,
            gamma=parser.args.gamma,
            horizon=self.INF,
            enable_pool=True,
        )
        data_size = parser.args.sampler_size
        sample_range = [int(x) for x in np.linspace(8, data_size, num_samples)]
        for dataset_index in env_indices:
            for algorithm in algorithms:
                for policy_index in policy_indices:
                    for trial_index in range(iid_resample_times):
                        multi_args.append(
                            (
                                trial_index,
                                algorithm,
                                policy_index,
                                candidates,
                                dataset_index,
                                data_size,
                                sample_range,
                            )
                        )
        if os.path.exists(f"offline_data/sample_eff_cache/results.pkl"):
            selection_results = self.toolbox.load_from_pkl(
                f"offline_data/sample_eff_cache/results"
            )
        else:
            num_cores = available_cpu_cores(percent=10, limit=20)
            with Pool(processes=num_cores) as pool:
                selection_results = list(
                    tqdm(
                        pool.imap(incremental_selection_as_single_task, multi_args),
                        total=len(multi_args),
                        desc="Observing sample efficiency...",
                    )
                )
                selection_results = [
                    [args, result]
                    for args, result in zip(multi_args, selection_results)
                ]
            os.makedirs(f"offline_data/sample_eff_cache/", exist_ok=True)
            self.toolbox.save_as_pkl(
                f"offline_data/sample_eff_cache/results", selection_results
            )
        pointer = 0
        for i, dataset_index in enumerate(env_indices):
            plt.figure(figsize=(9.5, 6))
            expectation = np.mean([j_table[dataset_index][j] for j in policy_indices])
            if len(policy_indices) == 1:
                normalizer = 1
            else:
                normalizer = math.sqrt(
                    np.mean(
                        [
                            (j_table[dataset_index][j] - expectation) ** 2
                            for j in policy_indices
                        ]
                    )
                )
            for alg in algorithms:
                curves_with_diff_estimates = [[] for _ in sample_range]
                for policy_index in policy_indices:
                    for _ in range(iid_resample_times):
                        _, rankings = selection_results[pointer]
                        pointer += 1
                        for j, env_hat in enumerate(rankings):
                            if alg == "trivial_random":
                                j_error = (
                                    np.mean(
                                        [
                                            math.fabs(
                                                j_table[random_choice][policy_index]
                                                - j_table[dataset_index][policy_index]
                                            )
                                            for random_choice in candidates
                                        ]
                                    )
                                    / normalizer
                                )
                            else:
                                j_error = math.fabs(
                                    j_table[env_hat][policy_index]
                                    - j_table[dataset_index][policy_index]
                                )
                                j_error /= normalizer
                            curves_with_diff_estimates[j].append(j_error)
                # print(curves_with_diff_policy)
                j_errors_mean, j_errors_low, j_errors_high = zip(
                    *[
                        stat_from_seq(j_errors, num_resamples=512, confidence=95)
                        for j_errors in curves_with_diff_estimates
                    ]
                )
                plt.plot(sample_range, j_errors_mean, label=f"{alg}")
                plt.fill_between(sample_range, j_errors_low, j_errors_high, alpha=0.2)
            plt.rcParams.update({"font.size": 8})
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.2, 1),
            )
            plt.xlabel(f"Resample Size")
            plt.ylabel(f"Normalized J error")
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            left_ticks = ax1.get_yticks()
            ax1.set_ylim(left_ticks[0], left_ticks[-1])
            right_ticks = left_ticks * normalizer
            ax2.set_yticks(right_ticks)
            ax2.set_yticklabels([f"{tick:.1f}" for tick in right_ticks])
            ax2.set_ylabel("Original J Error")
            ax2.set_ylim(right_ticks[0], right_ticks[-1])
            plt.title(
                f"Selection Results Under Different Sample Size\n"
                f"(with ground model selected as M_{dataset_index})"
            )
            plt.tight_layout()
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f"eff_{dataset_index}.pdf"))
            plt.close()

    def observe_mis_spec_level(
        self,
        num_levels,
        num_candidates,
        confidence,
        num_states,
        gamma,
        horizon,
        num_resamples,
    ):
        if os.path.exists(f"offline_data/figures/mis_spec/mis_spec.pdf"):
            return
        assert num_levels + num_candidates - 2 < len(parser.args.hopper_gravities)
        target = 0
        selector = SelectionBaselines()
        policy_indices = parser.args.policy_for_traverse
        j_table, l_table = self.pre_compute(num_states, gamma, horizon)
        max_size = parser.args.sampler_size
        multi_args = [
            (
                algorithm,
                policy_index,
                list(range(level, level + num_candidates)),
                target,
                max_size,
            )
            for algorithm in selector.baselines
            for level in range(num_levels)
            for policy_index in policy_indices
            for _ in range(num_resamples)
        ]
        num_cores = available_cpu_cores(percent=10, limit=24)
        with Pool(processes=num_cores) as pool:
            selection_results = list(
                tqdm(
                    pool.imap(selection_as_single_task, multi_args),
                    total=len(multi_args),
                    desc="Observing selections of different mis-specification levels...",
                )
            )
        count = 0
        for algorithm in selector.baselines:
            j_errors_spec_alg = []
            j_conf_low_spec_alg = []
            j_conf_high_spec_alg = []
            for level in range(num_levels):
                candidates = list(range(level, level + num_candidates))
                j_errors = []
                for policy_index in policy_indices:
                    for _ in range(num_resamples):
                        ranked_env_indices, loss = selection_results[count]
                        count += 1
                        env_hat = ranked_env_indices[0]
                        j_error = math.fabs(
                            j_table[env_hat][policy_index]
                            - j_table[target][policy_index]
                        )
                        j_errors.append(j_error)
                j_mean, j_low, j_high = stat_from_seq(
                    j_errors, confidence, num_resamples
                )
                j_errors_spec_alg.append(j_mean)
                j_conf_low_spec_alg.append(j_low)
                j_conf_high_spec_alg.append(j_high)
            plt.plot(list(range(num_levels)), j_errors_spec_alg, label=f"{algorithm}")
            plt.fill_between(
                list(range(num_levels)),
                j_conf_low_spec_alg,
                j_conf_high_spec_alg,
                alpha=0.2,
            )
        plt.legend()
        plt.xlabel("Mis-specification Level")
        plt.ylabel("Un-normalized J Error")
        plt.title(
            f"J Error in Fluctuation with Mis-specification Level\n"
            f"(with num_candidates={num_candidates}, confidence_level={confidence}, num_resamples={num_resamples})"
        )
        os.makedirs(f"offline_data/figures/mis_spec/", exist_ok=True)
        plt.savefig(f"offline_data/figures/mis_spec/mis_spec.pdf")
        plt.close()

    def afterward_observe(self):
        self.observe_bvft()
        self.observe_param_gap(
            num_states=1024,
            gamma=parser.args.gamma,
            horizon=self.INF,
            resample_size=2048,
            num_resamples=32,
            confidence=95,
        )
        self.observe_sample_efficiency(
            algorithms=model_free_alg
            if not parser.args.enable_bellman_operators
            else model_based_alg,
            path="offline_data/figures/sample_eff",
            num_init_states=1024,
            iid_resample_times=96,
            num_samples=8,
        )
        self.observe_mis_spec_level(
            num_levels=len(parser.args.hopper_gravities) - 4,
            num_candidates=5,
            confidence=95,
            num_states=1024,
            gamma=parser.args.gamma,
            horizon=self.INF,
            num_resamples=32,
        )
