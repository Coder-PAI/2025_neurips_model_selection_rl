import glob
import itertools
import os
import parser
import pickle
import re
import shutil
from collections.abc import Hashable, Iterable
from typing import Any, Dict, List, Tuple, Union

import gym
import gym.envs.mujoco.hopper_v4
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore, Style
from scipy.stats import trim_mean
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict(policy, x_in):
    x = torch.tensor(x_in, dtype=torch.float32, device=device)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return policy(x).detach().cpu().numpy()


def stat_from_seq(seq, confidence, num_resamples, trim_prop: float = 0, epsilon=1e-8):
    """
    Given a sequence consisting of n i.i.d. samples, estimate the sample mean and the statistical error
    of the estimated mean (by doing m times of resampling with replacement sized n).
    """
    # 0 <= confidence <= 100
    mean = trim_mean(seq, trim_prop)
    means_resample = [
        trim_mean(np.random.choice(seq, len(seq), replace=True), trim_prop)
        for _ in range(num_resamples)
    ]
    low_percentile = np.percentile(means_resample, (100 - confidence) / 2)
    high_percentile = np.percentile(means_resample, (100 + confidence) / 2)
    return mean, low_percentile - epsilon, high_percentile + epsilon


class StochasticEnv(gym.Env):
    def __init__(
        self,
        mujoco_env,
        force_mean,
        force_scaler,
    ):
        self.env = mujoco_env
        g_x, g_y, g_z = self.env.unwrapped.model.opt.gravity
        assert g_x == 0 and g_y == 0 and g_z < 0
        self.force_mean = force_mean
        self.force_scaler = force_scaler

    def step(self, action):
        force = np.random.normal(
            loc=self.force_mean, scale=self.force_scaler, size=(3,)
        )
        for i in range(self.env.model.nbody):
            self.env.data.xfrc_applied[i][:3] = force
        transition = list(self.env.step(action))
        transition[0] = np.array(transition[0])
        return tuple(transition)

    def reset(self, *, seed: int = None, options: dict = None) -> tuple:
        return self.env.reset()

    def __getattr__(self, item):
        attr = getattr(self.env, item)
        if callable(attr):
            return lambda *args, **kwargs: attr(*args, **kwargs)
        else:
            return attr


class GeneralUtils:
    def __init__(self, verbose=False):
        self.device = device
        self.verbose = verbose
        self.state_action_hash_index = None

    @staticmethod
    def create_placeholder(path):
        with open(path, "w") as _:
            pass

    @staticmethod
    def delete_bvft_cache():
        if os.path.exists(f"offline_data/bvft"):
            shutil.rmtree(f"offline_data/bvft")

    def remove_all_placeholders(self):
        (
            env_class,
            policy_class,
            _,
        ) = self.load_env_policy_dataset()
        dataset_indices = parser.args.ground_truth_for_traverse
        policy_indices = parser.args.policy_for_traverse
        for i in tqdm(range(len(env_class))):
            for j in policy_indices:
                for t in dataset_indices:
                    if os.path.exists(
                        f"offline_data/q_functions/{i}/{j}/{t}/{parser.args.placeholder}"
                    ):
                        os.remove(
                            f"offline_data/q_functions/{i}/{j}/{t}/{parser.args.placeholder}"
                        )
                    if os.path.exists(
                        f"offline_data/v_functions/{i}/{j}/{t}/{parser.args.placeholder}"
                    ):
                        os.remove(
                            f"offline_data/v_functions/{i}/{j}/{t}/{parser.args.placeholder}"
                        )
        for k in tqdm(range(len(env_class))):
            for i in range(len(env_class)):
                for j in policy_indices:
                    for t in dataset_indices:
                        if os.path.exists(
                            f"offline_data/bellman_operators/{k}/{i}/{j}/{t}/{parser.args.placeholder}"
                        ):
                            os.remove(
                                f"offline_data/bellman_operators/{k}/{i}/{j}/{t}/{parser.args.placeholder}"
                            )

    def preparations(
        self,
        policy_index: int,
        env_indices: List[int],
        dataset_index: int,
    ):
        # policy = self.load_policy(parser.args.trainer_algorithm, policy_index, device)

        # Prepare L different q functions
        q_cur_tables = [
            [
                q
                for trajectory in self.load_q_function(
                    env_index=i, policy_index=policy_index, dataset_index=dataset_index
                )
                for q in trajectory
            ]
            for i in env_indices
        ]

        v_next_tables = [
            [
                v
                for trajectory in self.load_v_function(
                    env_index=i, policy_index=policy_index, dataset_index=dataset_index
                )
                for v in trajectory[1:]
            ]
            for i in env_indices
        ]

        for q_cur_table, v_next_table in zip(q_cur_tables, v_next_tables):
            if len(q_cur_table) != len(v_next_table):
                raise AttributeError(
                    f"Data size mismatch! Q table sized {len(q_cur_table)}, V table sized "
                    f"{len(v_next_table)}"
                )

        dataset = self.load_dataset(dataset_index)

        return q_cur_tables, v_next_tables, dataset

    def load_ranking(
        self,
        selection_algorithm: str,
        dataset_index: int,
        policy_index: int,
        bool_mis_spec: bool,
    ):
        return (
            self.load_from_pkl(
                f"offline_data/realizable_selections/{selection_algorithm}/{dataset_index}/{policy_index}/ranking"
            )
            if not bool_mis_spec
            else self.load_from_pkl(
                f"offline_data/mis_spec_selections/{selection_algorithm}/{dataset_index}/{policy_index}/ranking"
            )
        )

    @staticmethod
    def clear_ckpt(index: int):
        ckpt_folder = f"offline_data/policies/{index}/DDPG_online"
        ckpt_to_delete = glob.glob(os.path.join(ckpt_folder, "*.pth"))
        for file_path in ckpt_to_delete:
            try:
                os.remove(file_path)
                print(f"Checkpoint Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    def env_copy(self, env, mode=None):
        env_params = self.hopper_get_params(env)
        if mode is None:
            return self.hopper_create_with_params(env_params)
        else:
            return self.hopper_create_with_params(env_params, mode)

    def render_dataset(self, dataset_index, num_frames):
        def save(path, dataset_index, episode_index, timestamps, frames):
            num_frames = len(frames)
            fig, axs = plt.subplots(1, num_frames, figsize=(8, 2))
            for t, frame, ax in zip(timestamps, frames, axs):
                ax.imshow(frame)
                ax.axis("off")
                ax.set_title(f"Frame {t}")
            plt.suptitle(f"Episode {episode_index} of the dataset {dataset_index}")
            plt.tight_layout()
            plt.savefig(path, format="pdf")
            plt.close()

        dataset = self.load_dataset(dataset_index)
        env = gym.make("Hopper-v4", render_mode="rgb_array")
        env.reset()
        mean_time = int(np.mean([trajectory["num_states"] for trajectory in dataset]))
        timestamps = [int(x) for x in np.linspace(0, mean_time - 1, num_frames)]
        if len(timestamps) > len(set(timestamps)):
            timestamps = list(range(mean_time))
        bar = tqdm(total=len(dataset) * len(timestamps), desc="Rendering Dataset...")
        for i, trajectory in enumerate(dataset):
            q_pos_list = trajectory["q_pos"]
            q_vel_list = trajectory["q_vel"]
            if trajectory["num_states"] < mean_time:
                bar.update(len(timestamps))
                continue
            frames = []
            for t in timestamps:
                bar.update(1)
                q_pos = q_pos_list[t]
                q_vel = q_vel_list[t]
                env.set_state(q_pos, q_vel)
                frame = env.render()
                frames.append(frame)
            os.makedirs(f"offline_data/figures/render/{dataset_index}", exist_ok=True)
            save(
                f"offline_data/figures/render/{dataset_index}/episode_{i}.pdf",
                dataset_index,
                i,
                timestamps,
                frames,
            )

    def render_by_pos_vel_list(self, env, traj: List[Tuple[np.ndarray, np.ndarray]]):
        env = self.env_copy(env, mode="rgb_array")
        env.reset()
        frames = []
        for q_pos, q_vel in traj:
            env.set_state(q_pos, q_vel)
            frames.append(env.render())
        return frames

    def compare_trajectory(self, env, traj_pred, traj_truth):
        print(self.hopper_get_params(env))
        for traj in [traj_pred, traj_truth]:
            self.render_by_pos_vel_list(env, traj)

    @staticmethod
    def q_pos_from_dataset(dataset, contain_last=True):
        return [
            trajectory["q_pos"][k]
            for trajectory in dataset
            for k in range(
                trajectory["num_states"]
                if contain_last
                else trajectory["num_states"] - 1
            )
        ]

    @staticmethod
    def q_vel_from_dataset(dataset, contain_last=True):
        return [
            trajectory["q_vel"][k]
            for trajectory in dataset
            for k in range(
                trajectory["num_states"]
                if contain_last
                else trajectory["num_states"] - 1
            )
        ]

    def hopper_render_offline_dataset(
        self,
        env: StochasticEnv,
        dataset,
        trajectory_num: Union[None, int] = None,
    ):
        env = self.env_copy(env, mode="human")
        self.render_text(str(self.hopper_get_params(env)), "YELLOW")
        for episode, trajectory in enumerate(
            dataset if trajectory_num is None else dataset[:trajectory_num]
        ):
            print(f"Replaying episode {episode}...")
            q_pos_list = trajectory["q_pos"]
            q_vel_list = trajectory["q_vel"]
            env.reset()
            for q_pos, q_vel in zip(q_pos_list, q_vel_list):
                env.set_state(q_pos, q_vel)
                env.render()

    def hopper_render_policy_on_given_env(
        self, env: StochasticEnv, policy, trajectory_num
    ):
        env_playable = self.hopper_create_with_params(
            self.hopper_get_params(env), mode="human"
        )
        for _ in range(trajectory_num):
            obs, info = env_playable.reset()
            while True:
                action = policy.predict(np.array([obs]))[0]
                obs, reward, done, _, _ = env_playable.step(action)
                if done:
                    break

    @staticmethod
    def env_get_reward(env, action, q_pos_pair, _forward_reward_weight=1.0):
        q_pos_before, q_pos_after = q_pos_pair
        x_position_before = q_pos_before[0]
        x_position_after = q_pos_after[0]
        x_velocity = (x_position_after - x_position_before) / env.dt

        ctrl_cost = env.control_cost(action)

        forward_reward = _forward_reward_weight * x_velocity
        healthy_reward = env.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        reward = rewards - costs
        return reward

    @staticmethod
    def sort_name_by_index(name):
        match = re.search(r"(\d+)", name)
        if match:
            return int(match.group(1)), name
        else:
            return float("inf"), name

    def fix_all(self):
        env_indices = list(range(len(parser.args.hopper_gravities)))
        policy_indices = parser.args.policy_for_traverse
        dataset_indices = parser.args.ground_truth_for_traverse
        for env_index in env_indices:
            for policy_index in policy_indices:
                j_folder = f"offline_data/j_tables/{env_index}/{policy_index}"
                j_stat_error_folder = (
                    f"offline_data/j_stat_errors/{env_index}/{policy_index}"
                )
                l_folder = f"offline_data/l_tables/{env_index}/{policy_index}"
                l_stat_error_folder = (
                    f"offline_data/l_stat_errors/{env_index}/{policy_index}"
                )
                self.fix_eof_error(j_folder)
                self.fix_eof_error(j_stat_error_folder)
                self.fix_eof_error(l_folder)
                self.fix_eof_error(l_stat_error_folder)
        for env_index in env_indices:
            for policy_index in policy_indices:
                for dataset_index in dataset_indices:
                    q_folder = f"offline_data/q_functions/{env_index}/{policy_index}/{dataset_index}"
                    v_folder = f"offline_data/v_functions/{env_index}/{policy_index}/{dataset_index}"
                    self.fix_eof_error(q_folder)
                    self.fix_eof_error(v_folder)
        if parser.args.enable_bellman_operators:
            for env_index_0 in env_indices:
                for env_index_1 in env_indices:
                    for policy_index in policy_indices:
                        for dataset_index in dataset_indices:
                            t_folder = f"offline_data/bellman_operators/{env_index_0}/{env_index_1}/{policy_index}/{dataset_index}"
                            self.fix_eof_error(t_folder)

    def fix_eof_error(self, folder):
        if not os.path.exists(folder):
            return
        for filename in sorted(
            os.listdir(folder),
            key=lambda x: self.sort_name_by_index(x),
        ):
            if filename.endswith(".pkl"):
                filepath = os.path.join(folder, filename)
                with open(filepath, "rb") as file:
                    try:
                        pickle.load(file)
                    except Exception as e:
                        print(f"{e} at {filepath}. Removing the corrupted data.")
                        os.remove(filepath)

    def merge_batch_q_functions(self, env_index: int, policy_index: int, dataset_index):
        if os.path.exists(
            f"offline_data/q_functions/{env_index}/{policy_index}/{dataset_index}/q_function.pkl"
        ):
            return
        merged_q = []
        q_function_folder = (
            f"offline_data/q_functions/{env_index}/{policy_index}/{dataset_index}"
        )
        for filename in sorted(
            os.listdir(q_function_folder),
            key=lambda x: self.sort_name_by_index(x),
        ):
            if filename.endswith(".pkl"):
                filepath = os.path.join(q_function_folder, filename)
                with open(filepath, "rb") as file:
                    data = pickle.load(file)
                    merged_q += data.tolist()
                os.remove(filepath)

        dataset = self.load_dataset(dataset_index)
        merged_q = self.align_q_with_datasets(merged_q, [dataset])[0]

        path = f"offline_data/q_functions/{env_index}/{policy_index}/{dataset_index}/"
        os.makedirs(path, exist_ok=True)
        self.save_as_pkl(path + "q_function", merged_q)

        return merged_q

    def merge_batch_bellman_operators(
        self, env_cand_index: int, env_q_index: int, policy_index: int, dataset_index
    ):
        if os.path.exists(
            f"offline_data/bellman_operators/{env_cand_index}/{env_q_index}/{policy_index}/{dataset_index}/bellman_operator.pkl"
        ):
            return
        merged_q = []
        q_function_folder = f"offline_data/bellman_operators/{env_cand_index}/{env_q_index}/{policy_index}/{dataset_index}/"
        for filename in sorted(
            os.listdir(q_function_folder),
            key=lambda x: self.sort_name_by_index(x),
        ):
            if filename.endswith(".pkl"):
                filepath = os.path.join(q_function_folder, filename)
                with open(filepath, "rb") as file:
                    data = pickle.load(file)
                    merged_q += data.tolist()
                os.remove(filepath)

        dataset = self.load_dataset(dataset_index)
        merged_q = self.align_q_with_datasets(merged_q, [dataset])[0]

        path = f"offline_data/bellman_operators/{env_cand_index}/{env_q_index}/{policy_index}/{dataset_index}/"
        os.makedirs(path, exist_ok=True)
        self.save_as_pkl(path + "bellman_operator", merged_q)
        return merged_q

    def merge_batch_v_functions(self, env_index: int, policy_index: int, dataset_index):
        if os.path.exists(
            f"offline_data/v_functions/{env_index}/{policy_index}/{dataset_index}/v_function.pkl"
        ):
            return
        merged_v = []
        v_function_folder = (
            f"offline_data/v_functions/{env_index}/{policy_index}/{dataset_index}"
        )
        for filename in sorted(
            os.listdir(v_function_folder),
            key=lambda x: self.sort_name_by_index(x),
        ):
            if filename.endswith(".pkl"):
                filepath = os.path.join(v_function_folder, filename)
                with open(filepath, "rb") as file:
                    data = pickle.load(file)
                    merged_v += data.tolist()
                os.remove(filepath)

        dataset = self.load_dataset(dataset_index)
        merged_v = self.align_v_with_datasets(merged_v, [dataset])[0]

        path = f"offline_data/v_functions/{env_index}/{policy_index}/{dataset_index}/"
        os.makedirs(path, exist_ok=True)
        self.save_as_pkl(path + "v_function", merged_v)
        return merged_v

    @staticmethod
    def rewards_from_dataset(dataset):
        return [
            trajectory["rewards"][k]
            for trajectory in dataset
            for k in range(trajectory["num_states"] - 1)
        ]

    @staticmethod
    def state_action_with_consecutive_state_from_dataset(
        dataset,
    ):
        return [
            (
                trajectory["observations"][k],
                trajectory["actions"][k],
                trajectory["observations"][k + 1],
            )
            for trajectory in dataset
            for k in range(trajectory["num_states"] - 1)
        ]

    @staticmethod
    def all_data_from_dataset(dataset):
        """
        Return a list of tuples consisting of: state; q_pos; q_vel; action; reward; next_state; next_q_pos; next_q_vel.
        """
        return [
            (
                trajectory["observations"][k],
                trajectory["q_pos"][k],
                trajectory["q_vel"][k],
                trajectory["actions"][k],
                trajectory["rewards"][k],
                trajectory["observations"][k + 1],
                trajectory["q_pos"][k + 1],
                trajectory["q_vel"][k + 1],
            )
            for trajectory in dataset
            for k in range(trajectory["num_states"] - 1)
        ]

    @staticmethod
    def states_with_dones_from_dataset(dataset):
        return [
            (
                trajectory["observations"][k],
                trajectory["dones"][k],
            )
            for trajectory in dataset
            for k in range(trajectory["num_states"])
        ]

    @staticmethod
    def state_action_reward_with_next_state_from_dataset(
        dataset,
    ):
        return [
            (
                trajectory["observations"][k],
                trajectory["actions"][k],
                trajectory["rewards"][k],
                trajectory["observations"][k + 1],
            )
            for trajectory in dataset
            for k in range(trajectory["num_states"] - 1)
        ]

    @staticmethod
    def states_from_dataset(dataset):
        return [
            trajectory["observations"][k]
            for trajectory in dataset
            for k in range(trajectory["num_states"])
        ]

    @staticmethod
    def state_action_pairs_from_dataset(dataset):
        return [
            (
                trajectory["observations"][k],
                trajectory["actions"][k],
            )
            for trajectory in dataset
            for k in range(trajectory["num_states"] - 1)
        ]

    def state_action_with_pos_vel_from_dataset(self, dataset):
        state_action_pairs = self.state_action_pairs_from_dataset(dataset)
        q_pos_list = self.q_pos_from_dataset(dataset, contain_last=False)
        q_vel_list = self.q_vel_from_dataset(dataset, contain_last=False)
        assert len(q_pos_list) == len(q_vel_list) == len(state_action_pairs)
        return list(zip(state_action_pairs, q_pos_list, q_vel_list))

    def load_bellman_operator(
        self,
        env_cand_index: int,
        env_q_index: int,
        policy_index: int,
        dataset_index,
        num_rollout=None,
    ):
        all_estimates = self.load_from_pkl(
            f"offline_data/bellman_operators/{env_cand_index}/{env_q_index}/{policy_index}/{dataset_index}/bellman_operator"
        )
        if num_rollout is None:
            num_rollout = parser.args.rollout_num_trajectories
        for i, episode in enumerate(all_estimates):
            all_estimates[i] = np.mean(
                np.array(episode)[:, :num_rollout], axis=1, keepdims=False
            ).tolist()
        return all_estimates

    def check_rollouts_completed(self):
        (
            env_class,
            policy_class,
            _,
        ) = self.load_env_policy_dataset()
        # Check q functions
        policy_indices = parser.args.policy_for_traverse
        dataset_indices = parser.args.ground_truth_for_traverse
        num_tasks = 0
        q_unfinished = []
        v_unfinished = []
        t_unfinished = []
        for i in range(len(env_class)):
            for j in policy_indices:
                for t in dataset_indices:
                    if not os.path.exists(
                        f"offline_data/q_functions/{i}/{j}/{t}/q_function.pkl"
                    ):
                        q_unfinished.append([i, j, t])
                    num_tasks += 1
        # Check v functions
        for i in range(len(env_class)):
            for j in policy_indices:
                for t in dataset_indices:
                    if not os.path.exists(
                        f"offline_data/v_functions/{i}/{j}/{t}/v_function.pkl"
                    ):
                        v_unfinished.append([i, j, t])
                    num_tasks += 1
        if parser.args.enable_bellman_operators:
            # Check Bellman Operators
            for k in range(len(env_class)):
                for i in range(len(env_class)):
                    for j in policy_indices:
                        for t in dataset_indices:
                            if not os.path.exists(
                                f"offline_data/bellman_operators/{k}/{i}/{j}/{t}/bellman_operator.pkl"
                            ):
                                t_unfinished.append([k, i, j, t])
                            num_tasks += 1
        _, unfinished_q_policies, _ = (
            zip(*q_unfinished) if q_unfinished else [(), (), ()]
        )
        _, unfinished_v_policies, _ = (
            zip(*v_unfinished) if v_unfinished else [(), (), ()]
        )
        _, _, unfinished_t_policies, _ = (
            zip(*t_unfinished) if t_unfinished else [(), (), (), ()]
        )
        unfinished_policies = set(
            unfinished_q_policies + unfinished_v_policies + unfinished_t_policies
        )
        parser.completed_policies = list(
            set(parser.args.policy_for_traverse) - unfinished_policies
        )
        rate = (
            1 - (len(q_unfinished) + len(v_unfinished) + len(t_unfinished)) / num_tasks
        )
        print(f"Rate of completion: {rate}")
        if rate >= 0.9:
            print(f"Unfinished Q: {q_unfinished}")
            print(f"Unfinished V: {v_unfinished}")
            print(f"Unfinished T: {t_unfinished}")
        return not q_unfinished and not v_unfinished and not t_unfinished

    @staticmethod
    def states_with_pos_vel_dones_from_dataset(dataset):
        return [
            (
                trajectory["observations"][k],
                trajectory["q_pos"][k],
                trajectory["q_vel"][k],
                trajectory["dones"][k],
            )
            for trajectory in dataset
            for k in range(trajectory["num_states"])
        ]

    def get_iterable_size(self, x):
        if not isinstance(x, Iterable):
            return 1
        return sum(self.get_iterable_size(y) for y in x)

    def count_iterable_zeros(self, x):
        if not isinstance(x, Iterable):
            return 1 if x == 0 else 0
        return sum(self.count_iterable_zeros(y) for y in x)

    def align_v_with_datasets(self, v, datasets):
        size_v = self.get_iterable_size(v) // parser.args.rollout_num_trajectories
        size_datasets = sum(
            trajectory["num_states"] for dataset in datasets for trajectory in dataset
        )
        if not size_v == size_datasets:
            raise ValueError(f"States sized {size_v} != Datasets sized {size_datasets}")
        aligned_v = []
        pointer = 0
        for dataset in datasets:
            aligned_v.append([])
            for trajectory in dataset:
                aligned_v[-1].append([])
                for _ in range(trajectory["num_states"]):
                    aligned_v[-1][-1].append(v[pointer])
                    pointer += 1
        return aligned_v

    def align_q_with_datasets(self, q, datasets):
        size_q = self.get_iterable_size(q) // parser.args.rollout_num_trajectories
        size_datasets = sum(
            trajectory["num_states"] - 1
            for dataset in datasets
            for trajectory in dataset
        )
        if not size_q == size_datasets:
            raise ValueError(f"States sized {size_q} != Datasets sized {size_datasets}")
        aligned_q = []
        pointer = 0
        for dataset in datasets:
            aligned_q.append([])
            for trajectory in dataset:
                aligned_q[-1].append([])
                for _ in range(trajectory["num_states"] - 1):
                    aligned_q[-1][-1].append(q[pointer])
                    pointer += 1
        return aligned_q

    def state_action_with_pos_vel_from_merged_datasets(self, datasets):
        return list(
            itertools.chain.from_iterable(
                [
                    self.state_action_with_pos_vel_from_dataset(dataset)
                    for dataset in datasets
                ]
            )
        )

    def all_data_from_datasets(self, datasets):
        return list(
            itertools.chain.from_iterable(
                [self.all_data_from_dataset(dataset) for dataset in datasets]
            )
        )

    def states_with_pos_vel_dones_from_merged_datasets(self, datasets):
        return list(
            itertools.chain.from_iterable(
                [
                    self.states_with_pos_vel_dones_from_dataset(dataset)
                    for dataset in datasets
                ]
            )
        )

    @staticmethod
    def actions_from_dataset(dataset):
        return [
            trajectory["actions"][k]
            for trajectory in dataset
            for k in range(trajectory["num_states"] - 1)
        ]

    def v_function(
        self,
        v_table: Dict[Hashable, float],
        state: np.ndarray,
    ) -> float:
        return v_table[self.get_hashable(state)]

    def q_function(
        self,
        q_table: Dict[Tuple[Hashable, Hashable], float],
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        return q_table[
            (
                self.get_hashable(state),
                self.get_hashable(action),
            )
        ]

    def load_q_function(
        self, env_index: int, policy_index: int, dataset_index, num_rollout=None
    ):
        all_estimates = self.load_from_pkl(
            f"offline_data/q_functions/{env_index}/{policy_index}/{dataset_index}/q_function"
        )
        if num_rollout is None:
            num_rollout = parser.args.rollout_num_trajectories
        for i, episode in enumerate(all_estimates):
            all_estimates[i] = np.mean(
                np.array(episode)[:, :num_rollout], axis=1, keepdims=False
            ).tolist()
        return all_estimates

    def load_v_function(
        self, env_index: int, policy_index: int, dataset_index, num_rollout=None
    ):
        all_estimates = self.load_from_pkl(
            f"offline_data/v_functions/{env_index}/{policy_index}/{dataset_index}/v_function"
        )
        if num_rollout is None:
            num_rollout = parser.args.rollout_num_trajectories
        for i, episode in enumerate(all_estimates):
            all_estimates[i] = np.mean(
                np.array(episode)[:, :num_rollout], axis=1, keepdims=False
            ).tolist()
        return all_estimates

    def state_action_pairs_from_merged_datasets(self, datasets):
        return list(
            itertools.chain.from_iterable(
                [self.state_action_pairs_from_dataset(dataset) for dataset in datasets]
            )
        )

    def q_pos_list_from_merged_datasets(self, datasets):
        return list(
            itertools.chain.from_iterable(
                [self.q_pos_from_dataset(dataset) for dataset in datasets]
            )
        )

    def q_vel_list_from_merged_datasets(self, datasets):
        return list(
            itertools.chain.from_iterable(
                [self.q_vel_from_dataset(dataset) for dataset in datasets]
            )
        )

    @staticmethod
    def create_batch(list_data: List, batch_size: int):
        low_index = 0
        while low_index < len(list_data):
            high_index = min(len(list_data), low_index + batch_size)
            yield list_data[low_index:high_index].copy()
            low_index += batch_size

    def states_from_merged_datasets(self, datasets):
        return list(
            itertools.chain.from_iterable(
                [self.states_from_dataset(dataset) for dataset in datasets]
            )
        )

    @staticmethod
    def get_hashable(array) -> Hashable:
        return np.array2string(
            np.array(array),
            precision=16,
            formatter={"float_kind": lambda x: f"{x:.8f}"},
        )

    def load_env(self, env_index, mode=None):
        _, params = self.load_from_pkl(
            "offline_data/table_mdp/index_to_name_and_params"
        )[env_index]
        return self.hopper_create_with_params(params, mode)

    def load_env_policy_dataset(
        self, mode=None
    ) -> Tuple[List[StochasticEnv], List[Any], List[List[Dict[str, List]]],]:
        env_class = [
            self.hopper_create_with_params(mdp_env, mode)
            for _, mdp_env in self.load_from_pkl(
                "offline_data/table_mdp/index_to_name_and_params"
            ).values()
        ]
        policy_class = [
            self.load_policy(parser.args.trainer_algorithm, index, device)
            for index in range(len(env_class))
        ]
        datasets = [self.load_dataset(index) for index in range(len(env_class))]
        return env_class, policy_class, datasets

    def example_env_policy(self):
        index_to_name_and_params = self.load_from_pkl(
            "offline_data/table_mdp/index_to_name_and_params"
        )
        return index_to_name_and_params[0][1], self.load_policy(
            parser.args.trainer_algorithm, 0, device
        )

    @staticmethod
    def hopper_get_params(hopper_env):
        return {
            "gravity": hopper_env.unwrapped.model.opt.gravity,
            "force_mean": hopper_env.force_mean,
            "force_scaler": hopper_env.force_scaler,
        }

    @staticmethod
    def hopper_set_params(hopper_env, params):
        (
            gravity,
            force_mean,
            force_scaler,
        ) = params.values()
        env_with_noise = StochasticEnv(hopper_env, force_mean, force_scaler)
        env_with_noise.unwrapped.model.opt.gravity = gravity
        g_x, g_y, g_z = env_with_noise.unwrapped.model.opt.gravity
        assert g_x == 0 and g_y == 0 and g_z < 0
        return env_with_noise

    def hopper_create_with_params(self, params, mode=None):
        hopper_env = (
            gym.make(
                "Hopper-v4",
                render_mode=mode,
            )
            if mode is not None
            else gym.make(
                "Hopper-v4",
            )
        )
        return self.hopper_set_params(hopper_env, params)

    def load_env_class(self):
        return self.load_from_pkl(
            "offline_data/table_mdp/index_to_name_and_params"
        ), self.load_from_pkl("offline_data/table_mdp/mdp_name_to_index")

    @staticmethod
    def load_policy(algorithm_name: str, index: int, _device: str):
        policy = torch.load(f"offline_data/policies/{index}/{algorithm_name}.pth")
        policy.predict = lambda x: predict(policy=policy, x_in=x)
        return policy.to(device)

    def load_dataset(self, index: int):
        return self.load_from_pkl(f"offline_data/datasets/{index}/dataset")

    @staticmethod
    def flatten_from_datasets(estimations):
        return [
            x for dataset in estimations for trajectory in dataset for x in trajectory
        ]

    @staticmethod
    def v_from_dataset(estimations, initial_only=True):
        if initial_only:
            return [trajectory[0] for dataset in estimations for trajectory in dataset]
        else:
            return [
                v
                for dataset in estimations
                for trajectory in dataset
                for v in trajectory
            ]

    @staticmethod
    def render_text(text, color):
        print("\n" + getattr(Fore, color) + text + Style.RESET_ALL)

    @staticmethod
    def save_as_pkl(file_path, list_to_save):
        full_path = f"{file_path}.pkl"
        with open(full_path, "wb") as file:
            pickle.dump(list_to_save, file)

    @staticmethod
    def load_from_pkl(file_path):
        full_path = f"{file_path}.pkl"
        with open(full_path, "rb") as file:
            data = pickle.load(file)
        return data


__all__ = ["device", "GeneralUtils", "StochasticEnv"]
