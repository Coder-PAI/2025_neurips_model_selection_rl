import os
import parser
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from behaviour_policy import EpsilonGreedyPolicy
from global_utils import GeneralUtils, StochasticEnv


class OfflineDataSetCollector:

    def __init__(
            self,
            index_to_name_and_params: Dict[int, Tuple[str, StochasticEnv]],
            device,
            trajectory_max_horizon: int,
            data_size: int,
            epsilon: int,
            noise_scaler: int,
            verbose=True,
    ):
        self.trajectory_max_horizon = trajectory_max_horizon
        self.index_to_name_and_params = index_to_name_and_params
        self.toolbox = GeneralUtils()
        self.device = device
        self.verbose = verbose
        self.data_size = data_size

        # Epsilon-greedy sampler
        self.policy_class = {
            index: EpsilonGreedyPolicy(
                self.toolbox.load_policy(
                    algorithm_name=parser.args.trainer_algorithm,
                    index=index,
                    _device=device,
                ),
                epsilon=epsilon,
                noise_scaler=noise_scaler,
            )
            for index in self.index_to_name_and_params.keys()
        }

    def epsilon_greedy_sampler_over_mdp_class(self):
        self.toolbox.render_text("Start sampling offline dataset...", color="YELLOW")
        for index in self.index_to_name_and_params.keys():
            self.epsilon_greedy_sampler_for_single_mdp(index=index)
        self.toolbox.render_text(
            "Finished sampling offline dataset...",
            color="YELLOW",
        )

    def epsilon_greedy_sampler_for_single_mdp(self, index: int):
        if os.path.exists(f"offline_data/datasets/{index}/dataset.pkl"):
            self.toolbox.render_text(
                f"\tDataset {index} already exists.",
                color="RED",
            )
            return self.toolbox.load_dataset(index)
        mdp_env = self.toolbox.hopper_create_with_params(
            self.index_to_name_and_params[index][1]
        )
        # print(self.toolbox.hopper_get_params(mdp_env))
        policy = self.policy_class[index]
        (
            offline_dataset,
            num_state_action_pairs,
        ) = self.generate_trajectories(
            env=mdp_env, data_size=self.data_size, policy=policy
        )
        os.makedirs(f"offline_data/datasets/{index}", exist_ok=True)
        self.toolbox.save_as_pkl(
            f"offline_data/datasets/{index}/dataset",
            offline_dataset,
        )

    def target_policy_sampler_over_mdp_class(self, target_policy_index):
        self.toolbox.render_text("Start sampling offline dataset...", color="YELLOW")
        for index in self.index_to_name_and_params.keys():
            self.target_policy_sampler_for_single_mdp(
                index=index, target_policy_index=target_policy_index
            )
        self.toolbox.render_text(
            "Finished sampling offline dataset...",
            color="YELLOW",
        )

    def target_policy_sampler_for_single_mdp(
            self, index: int, target_policy_index: int
    ):
        if os.path.exists(f"offline_data/datasets/{index}/dataset.pkl"):
            self.toolbox.render_text(
                f"\tDataset {index} already exists.",
                color="RED",
            )
            return self.toolbox.load_dataset(index)
        mdp_env = self.toolbox.hopper_create_with_params(
            self.index_to_name_and_params[index][1]
        )
        # print(self.toolbox.hopper_get_params(mdp_env))
        policy = self.toolbox.load_policy("DDPG", target_policy_index, self.device)
        (
            offline_dataset,
            num_state_action_pairs,
        ) = self.generate_trajectories(
            env=mdp_env, data_size=parser.args.sampler_size, policy=policy
        )
        os.makedirs(f"offline_data/datasets/{index}", exist_ok=True)
        self.toolbox.save_as_pkl(
            f"offline_data/datasets/{index}/dataset",
            offline_dataset,
        )

    def generate_trajectories(
            self, env: StochasticEnv, data_size: int, policy: Any
    ) -> Tuple[List[Dict[str, Any]], int]:
        trajectories = []
        num_state_action_pairs = 0
        bar = tqdm(total=data_size, desc=f"Sampling dataset...")
        while num_state_action_pairs < data_size:
            episode = self.generate_single_trajectory(env, policy, bar)
            increment = episode["num_states"] - 1
            trajectories.append(episode)
            num_state_action_pairs += increment
        return trajectories, num_state_action_pairs

    def generate_single_trajectory(
            self, env: StochasticEnv, policy: Any, bar
    ) -> Dict[str, Any]:
        obs, _ = env.reset()

        observations = [obs]
        q_pos_list = [env.data.qpos.copy()]
        q_vel_list = [env.data.qvel.copy()]
        rewards = []
        actions = []
        dones = [False]
        num_states = 1

        for _ in range(self.trajectory_max_horizon):
            if dones[-1]:
                break
            bar.update(1)
            num_states += 1
            # physical simulation
            action = policy.predict(np.array(observations[-1], dtype=np.float32))[0]
            next_obs, reward, done, _, _ = env.step(action)

            # collect data
            q_pos_list.append(env.data.qpos.copy())
            q_vel_list.append(env.data.qvel.copy())
            observations.append(next_obs)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)

        episode_data = {
            "observations": observations,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
            "num_states": num_states,
            "q_pos": q_pos_list,
            "q_vel": q_vel_list,
        }
        return episode_data
