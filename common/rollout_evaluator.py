import os
import parser
import subprocess
import time
from functools import partial
from typing import Any, List

import numpy as np
from scipy.stats import trim_mean
from tqdm import tqdm

array64 = partial(np.array, dtype=np.float32)

from global_utils import GeneralUtils, StochasticEnv, device


class RolloutEvaluator:
    def __init__(
            self,
            gamma,
            attempts=30,
            max_horizon=100,
            verbose=True,
    ):
        self.toolbox = GeneralUtils()
        self.verbose = verbose
        self.gamma = gamma
        self.attempts = attempts
        self.max_horizon = max_horizon
        self.placeholder = parser.args.placeholder

    def start_rollout_on_single_session(self):
        self.toolbox.render_text(
            f"Embarrassingly Parallelism is forbidden. Start validation on 1 sessions...",
            color="YELLOW",
        )

        env_class, _, _ = self.toolbox.load_env_policy_dataset()
        dataset_indices = parser.args.ground_truth_for_traverse
        for policy_index in parser.args.policy_for_traverse:
            # Collect q function estimations for state-action pairs
            # (where the given action follows behaviour policy in the offline dataset)
            self.q_rollouts_for_all_datasets(env_class, policy_index, dataset_indices)

            # Collect v function estimations for states
            # (where all the actions in the trajectory follow the given policy indexed
            # as j)
            self.v_rollouts_for_all_datasets(env_class, policy_index, dataset_indices)

            # Collect bellman operators
            if parser.args.enable_bellman_operators:
                self.bellman_operator_for_everything(
                    env_class, policy_index, dataset_indices
                )

    def start_rollouts(self):
        self.toolbox.remove_all_placeholders()
        if parser.args.enable_embarrassingly:
            # Running RolloutEvaluator by Embarrassingly Parallelism
            self.toolbox.render_text(
                f"Embarrassingly Parallelism is enabled. Start validation on {parser.args.sessions} sessions...",
                color="YELLOW",
            )
            rollouts = []
            for i in range(parser.args.sessions):
                if parser.args.parallel_terminals_on:
                    process = subprocess.Popen(
                        [
                            "gnome-terminal",
                            "--",
                            "bash",
                            "-c",
                            "python3 rollout_evaluator.py",
                        ]
                        + [f"--{arg} {val}" for arg, val in vars(parser.args).items()],
                    )
                else:
                    process = subprocess.Popen(
                        ["python3", "rollout_evaluator.py"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                rollouts.append(process)
                time.sleep(0.1)

            for process in rollouts:
                process.wait()
        else:
            self.start_rollout_on_single_session()

    def q_rollouts_for_all_datasets(
            self,
            env_class: List[StochasticEnv],
            policy_index,
            dataset_indices,
    ):
        datasets = [self.toolbox.load_dataset(index) for index in dataset_indices]
        policy = self.toolbox.load_policy(
            index=policy_index,
            _device=device,
            algorithm_name=parser.args.trainer_algorithm,
        )
        for i, env in enumerate(env_class):
            for j, dataset in zip(dataset_indices, datasets):
                all_data_for_q = (
                    self.toolbox.state_action_with_pos_vel_from_merged_datasets(
                        [dataset]
                    )
                )
                # #input(all_data_for_q)
                os.makedirs(
                    f"offline_data/q_functions/{i}/{policy_index}/{j}/",
                    exist_ok=True,
                )
                if os.path.exists(
                        f"offline_data/q_functions/{i}/{policy_index}/{j}/{self.placeholder}"
                ):
                    self.toolbox.render_text(
                        f"\tQ function {(i, policy_index)} at dataset {j} is under processing by another session... Skipping.",
                        "BLUE",
                    )
                    continue
                self.toolbox.create_placeholder(
                    f"offline_data/q_functions/{i}/{policy_index}/{j}/{self.placeholder}"
                )
                for batch_id, batch_data in enumerate(
                        self.toolbox.create_batch(
                            all_data_for_q,
                            batch_size=parser.args.rollout_batch_size,
                        )
                ):
                    (
                        batch_state_action_pairs,
                        batch_q_pos,
                        batch_q_vel,
                    ) = zip(*batch_data)
                    q_exists = self.q_evaluate_given_env_policy_nn_batch_inference(
                        env,
                        policy,
                        batch_state_action_pairs,
                        batch_q_pos,
                        batch_q_vel,
                        i,
                        policy_index,
                        j,
                        batch_id,
                    )
                    if q_exists:
                        break
                self.toolbox.merge_batch_q_functions(
                    env_index=i, policy_index=policy_index, dataset_index=j
                )
                os.remove(
                    f"offline_data/q_functions/{i}/{policy_index}/{j}/{self.placeholder}"
                )

    def v_rollouts_for_all_datasets(
            self,
            env_class: List[StochasticEnv],
            policy_index,
            dataset_indices,
    ):
        datasets = [self.toolbox.load_dataset(index) for index in dataset_indices]
        policy = self.toolbox.load_policy(
            index=policy_index,
            _device=device,
            algorithm_name=parser.args.trainer_algorithm,
        )
        for i, env in enumerate(env_class):
            for j, dataset in zip(dataset_indices, datasets):
                all_data_for_v = (
                    self.toolbox.states_with_pos_vel_dones_from_merged_datasets(
                        [dataset]
                    )
                )
                # #input(all_data_for_v)
                os.makedirs(
                    f"offline_data/v_functions/{i}/{policy_index}/{j}/",
                    exist_ok=True,
                )
                if os.path.exists(
                        f"offline_data/v_functions/{i}/{policy_index}/{j}/{self.placeholder}"
                ):
                    self.toolbox.render_text(
                        f"\tV function {(i, policy_index)} at dataset {j} is under processing by another session... Skipping.",
                        "BLUE",
                    )
                    continue
                self.toolbox.create_placeholder(
                    f"offline_data/v_functions/{i}/{policy_index}/{j}/{self.placeholder}"
                )
                for batch_id, batch_data in enumerate(
                        self.toolbox.create_batch(
                            all_data_for_v,
                            batch_size=parser.args.rollout_batch_size,
                        )
                ):
                    (
                        batch_states,
                        batch_q_pos,
                        batch_q_vel,
                        batch_dones,
                    ) = zip(*batch_data)
                    v_exists = self.v_evaluate_given_env_policy_nn_batch_inference(
                        env,
                        policy,
                        batch_states,
                        batch_q_pos,
                        batch_q_vel,
                        batch_dones,
                        i,
                        policy_index,
                        j,
                        batch_id,
                    )
                    if v_exists:
                        break
                self.toolbox.merge_batch_v_functions(
                    env_index=i, policy_index=policy_index, dataset_index=j
                )
                os.remove(
                    f"offline_data/v_functions/{i}/{policy_index}/{j}/{self.placeholder}"
                )

    def v_evaluate_given_env_policy_nn_batch_inference(
            self,
            env,
            policy,
            states,
            q_pos_list,
            q_vel_list,
            dones,
            i,
            j,
            dataset_index,
            batch_id,
            trim_prop=0.1,
    ):
        os.makedirs(f"offline_data/v_functions/{i}/{j}/{dataset_index}", exist_ok=True)
        if os.path.exists(
                f"offline_data/v_functions/{i}/{j}/{dataset_index}/v_function.pkl"
        ):
            self.toolbox.render_text(
                f"\tV function {(i, j)} at dataset {dataset_index} already exists.",
                color="RED",
            )
            return True
        if os.path.exists(
                f"offline_data/v_functions/{i}/{j}/{dataset_index}/v_function_{batch_id}.pkl"
        ):
            self.toolbox.render_text(
                f"\tV function {(i, j)} at dataset {dataset_index}, batch {batch_id} already exists.",
                color="RED",
            )
            return False

        assert len(states) == len(q_pos_list) == len(q_vel_list) == len(dones)

        v_estimations = trim_mean(
            array64(
                [
                    self.batch_simulations(
                        env, policy, states, None, q_pos_list, q_vel_list, dones, "V"
                    )
                    for _ in range(parser.args.rollout_num_trajectories)
                ]
            ),
            proportiontocut=trim_prop,
            axis=0,
        )

        self.toolbox.save_as_pkl(
            f"offline_data/v_functions/{i}/{j}/{dataset_index}/v_function_{batch_id}",
            list(v_estimations),
        )

    def q_evaluate_given_env_policy_nn_batch_inference(
            self,
            env,
            policy,
            state_action_pairs,
            q_pos_list,
            q_vel_list,
            i,
            j,
            k,
            batch_id,
            trim_prop=0.1,
    ):
        os.makedirs(f"offline_data/q_functions/{i}/{j}/{k}", exist_ok=True)
        if os.path.exists(f"offline_data/q_functions/{i}/{j}/{k}/q_function.pkl"):
            self.toolbox.render_text(
                f"\tQ function {(i, j)} at dataset {k} already exists.",
                color="RED",
            )
            return True
        if os.path.exists(
                f"offline_data/q_functions/{i}/{j}/{k}/q_function_{batch_id}.pkl"
        ):
            self.toolbox.render_text(
                f"\tQ function {(i, j)} at dataset {k}, batch {batch_id} already exists.",
                color="RED",
            )
            return False

        assert len(state_action_pairs) == len(q_pos_list) == len(q_vel_list)
        states, actions = zip(*state_action_pairs)

        qq = array64(
            [
                self.batch_simulations(
                    env,
                    policy,
                    states,
                    actions,
                    q_pos_list,
                    q_vel_list,
                    [False for _ in state_action_pairs],
                    "Q",
                )
                for _ in range(parser.args.rollout_num_trajectories)
            ]
        )

        q_estimations = trim_mean(qq, proportiontocut=trim_prop, axis=0)

        self.toolbox.save_as_pkl(
            f"offline_data/q_functions/{i}/{j}/{k}/q_function_{batch_id}",
            list(q_estimations),
        )

    @staticmethod
    def forward(env: StochasticEnv, q_pos, q_vel, action):
        env.reset()
        # specify the current state
        env.set_state(q_pos, q_vel)
        # take action
        next_obs, reward, done, _, _ = env.step(action)
        q_pos = env.data.qpos.copy()
        q_vel = env.data.qvel.copy()
        return next_obs, reward, done, q_pos, q_vel

    def batch_simulations_debug(
            self,
            env: StochasticEnv,
            policy: Any,
            states: List[np.ndarray],
            actions,
            q_pos_list,
            q_vel_list,
            dones,
            mode,
            episodes,
    ):
        assert mode in ["V", "Q"]
        assert (mode == "V" and actions is None) or (
                mode == "Q" and actions is not None
        )
        max_horizon = self.max_horizon - 1 if mode == "V" else self.max_horizon

        self.toolbox.render_text(
            f"Start evaluating {mode} functions for given states...",
            color="YELLOW",
        )

        env_params = self.toolbox.hopper_get_params(env)

        env_copy = self.toolbox.hopper_create_with_params(env_params)
        # initial states
        reset_buffer = [(q_pos, q_vel) for q_pos, q_vel in zip(q_pos_list, q_vel_list)]
        done_buffer = [done for done in dones]
        state_buffer = [state for state in states]
        return_buffer = [0.0 for _ in states]

        # initial actions
        batch_states = array64(state_buffer)
        batch_actions = actions if actions is not None else policy.predict(batch_states)

        rewards = [None for _ in states]

        # Batch inference and sampling
        for k in range(max_horizon):
            if all(done_buffer):
                break
            for i in range(len(states)):
                if done_buffer[i]:
                    continue
                q_pos, q_vel = reset_buffer[i]
                # #input(f"q_pos: {q_pos}")
                # #input(f"q_vel: {q_vel}")
                # #input(f"action: {batch_actions[i]}")
                next_obs, reward, done, q_pos, q_vel = self.forward(
                    env_copy, q_pos, q_vel, batch_actions[i]
                )
                rewards[i] = reward
                # #input(f"reward:{reward}")
                reset_buffer[i] = (q_pos, q_vel)
                return_buffer[i] += self.gamma ** k * reward
                done_buffer[i] = done
                state_buffer[i] = next_obs
                # print(f"\n\n\n")
            # print(f"step {k}, sim:\t{rewards}")
            x = [episode["rewards"][k] for episode in episodes]
            # input(f"step {k}, dataset: \t{x}")
            batch_states = array64(state_buffer)
            batch_actions = policy.predict(batch_states)

    def batch_simulations(
            self,
            env: StochasticEnv,
            policy: Any,
            states: List[np.array],
            actions,
            q_pos_list,
            q_vel_list,
            dones,
            mode,
    ):
        assert mode in ["V", "Q"]
        assert (mode == "V" and actions is None) or (
                mode == "Q" and actions is not None
        )
        max_horizon = self.max_horizon - 1 if mode == "V" else self.max_horizon

        self.toolbox.render_text(
            f"Start evaluating {mode} functions for given states...",
            color="YELLOW",
        )

        env_params = self.toolbox.hopper_get_params(env)

        env_copy = self.toolbox.hopper_create_with_params(env_params)
        env_copy.reset()
        # initial states
        reset_buffer = [(q_pos, q_vel) for q_pos, q_vel in zip(q_pos_list, q_vel_list)]
        done_buffer = [done for done in dones]
        state_buffer = [state for state in states]
        return_buffer = [0.0 for _ in states]

        if all(done_buffer):
            return return_buffer

        # initial actions
        # print(state_buffer)
        batch_indices = [index for index, done in enumerate(done_buffer) if not done]
        batch_states = np.array([state_buffer[index] for index in batch_indices])
        if actions is not None:
            batch_actions = np.array([actions[index] for index in batch_indices])
        else:
            batch_actions = policy.predict(batch_states)
        assert len(batch_states) == len(batch_actions) == len(batch_indices)

        # input(f"state={states[3]}, actions={actions[3]}")

        debug_buffer = [[] for _ in states]

        # Batch inference and sampling
        for k in tqdm(range(max_horizon), desc="Sampling Trajectories"):
            for i, index in enumerate(batch_indices):
                q_pos, q_vel = reset_buffer[index]
                # #input(f"q_pos: {q_pos}")
                # #input(f"q_vel: {q_vel}")
                # #input(f"action: {batch_actions[i]}")
                env_copy.set_state(q_pos, q_vel)
                next_obs, reward, done, _, _ = env_copy.step(batch_actions[i])
                q_pos = env_copy.data.qpos.copy()
                q_vel = env_copy.data.qvel.copy()
                # debug_buffer[i].append([batch_states[i].copy()[:3], batch_actions[i].copy(), reward, next_obs.copy()])
                # #input(f"reward:{reward}")
                reset_buffer[index] = (q_pos.copy(), q_vel.copy())
                return_buffer[index] += self.gamma ** k * reward
                # #input(f"return:{return_buffer[i]}")
                done_buffer[index] = done
                state_buffer[index] = next_obs
                # print(f"\n\n\n")
            if all(done_buffer):
                break
            batch_indices = [
                index for index, done in enumerate(done_buffer) if not done
            ]
            batch_states = np.array([state_buffer[index] for index in batch_indices])
            batch_actions = policy.predict(batch_states)
            assert len(batch_states) == len(batch_actions) == len(batch_indices)
            # assert np.array_equal(x_actions, batch_actions)

        self.toolbox.render_text(
            f"Finished evaluating {mode} functions for given states.",
            color="YELLOW",
        )
        return return_buffer

    def batch_bellman_operator(
            self,
            env_cand,
            env_q,
            policy,
            batch_states,
            batch_q_pos,
            batch_q_vel,
            batch_actions,
            num_samples=parser.args.rollout_num_trajectories,
            trim_prop=0.1,
    ):

        assert (
                len(batch_states)
                == len(batch_q_pos)
                == len(batch_q_vel)
                == len(batch_actions)
        )
        batch_outputs = []
        for _ in range(num_samples):
            batch_next_states = []
            batch_next_q_pos = []
            batch_next_q_vel = []
            batch_rewards = []
            batch_dones = []

            # Collect next states under env_cand
            for q_pos, q_vel, action in zip(batch_q_pos, batch_q_vel, batch_actions):
                env_cand.reset()
                env_cand.set_state(q_pos, q_vel)
                next_state, reward, done, _, _ = env_cand.step(action)
                batch_next_states.append(next_state)
                batch_next_q_pos.append(env_cand.data.qpos.copy())
                batch_next_q_vel.append(env_cand.data.qvel.copy())
                batch_rewards.append(reward)
                batch_dones.append(done)

            # Simulate Q_{env_q}(next_state, policy)=V_{env_q}^{policy}(next_state) under env_q
            v_simulations = self.batch_simulations(
                env_q,
                policy,
                batch_next_states,
                None,
                batch_next_q_pos,
                batch_next_q_vel,
                batch_dones,
                "V",
            )
            # Update batch outputs
            batch_outputs.append(
                [
                    reward + parser.args.gamma * v
                    for reward, v in zip(batch_rewards, v_simulations)
                ]
            )
        # Return the outputs on average
        return trim_mean(np.array(batch_outputs), proportiontocut=trim_prop, axis=0)

    def bellman_operator(self, env_cand, env_q, policy, all_data):
        all_outputs = []
        for batch_data in self.toolbox.create_batch(
                all_data,
                batch_size=parser.args.model_based_batch_size,
        ):
            batch_state_actions, batch_q_pos, batch_q_vel = zip(*batch_data)
            batch_states, batch_actions = zip(*batch_state_actions)
            batch_output = self.batch_bellman_operator(
                env_cand,
                env_q,
                policy,
                batch_states,
                batch_q_pos,
                batch_q_vel,
                batch_actions,
            )
            all_outputs += list(batch_output)
        return all_outputs

    def bellman_operator_for_everything(
            self,
            env_class: List[StochasticEnv],
            policy_index,
            dataset_indices,
    ):
        """
        The function is responsible for estimating T_{M_k}Q_{M_i}(pi_j) by random sampling.
        """
        datasets = [self.toolbox.load_dataset(index) for index in dataset_indices]
        policy = self.toolbox.load_policy(
            index=policy_index,
            _device=device,
            algorithm_name=parser.args.trainer_algorithm,
        )
        for k, env_cand in enumerate(env_class):  # T_{M_k}
            for i, env_q in enumerate(env_class):  # M_i
                for j, dataset in zip(dataset_indices, datasets):  # pi_j
                    # Check if there's another session processing Bellman Operator (k, i, policy_index, j)
                    os.makedirs(
                        f"offline_data/bellman_operators/{k}/{i}/{policy_index}/{j}",
                        exist_ok=True,
                    )
                    if os.path.exists(
                            f"offline_data/bellman_operators/{k}/{i}/{policy_index}/{j}/{self.placeholder}"
                    ):
                        self.toolbox.render_text(
                            f"\tBellman Operator {(k, i, policy_index)} at dataset {j} is under processing by another session... Skipping.",
                            "BLUE",
                        )
                        continue
                    self.toolbox.create_placeholder(
                        f"offline_data/bellman_operators/{k}/{i}/{policy_index}/{j}/{self.placeholder}"
                    )

                    # If not, a placeholder will be placed under the corresponding folder (k, i, j) and the evaluation
                    # process would commence.
                    all_data_for_q = (
                        self.toolbox.state_action_with_pos_vel_from_merged_datasets(
                            [dataset]
                        )
                    )
                    for batch_id, batch_data in enumerate(
                            self.toolbox.create_batch(
                                all_data_for_q,
                                batch_size=parser.args.model_based_batch_size,
                            )
                    ):
                        # Check if the final output at (k, i, j) has already existed
                        if os.path.exists(
                                f"offline_data/bellman_operators/{k}/{i}/{policy_index}/{j}/bellman_operator.pkl"
                        ):
                            self.toolbox.render_text(
                                f"\tBellman Operator at {(k, i, policy_index)} at dataset {j} already exists.",
                                color="RED",
                            )
                            break
                        # Check if the output at batch "batch_id" has already existed
                        elif os.path.exists(
                                f"offline_data/bellman_operators/{k}/{i}/{policy_index}/{j}/bellman_operator_{batch_id}.pkl"
                        ):
                            self.toolbox.render_text(
                                f"\tBellman Operator {(k, i, policy_index)} at dataset {j}, batch {batch_id} already exists.",
                                color="RED",
                            )
                            continue
                        output = self.bellman_operator(
                            env_cand, env_q, policy, batch_data
                        )
                        self.toolbox.save_as_pkl(
                            f"offline_data/bellman_operators/{k}/{i}/{policy_index}/{j}/bellman_operator_{batch_id}",
                            output,
                        )

                    # Merge batch output for the bellman operator at (k, i, j)
                    self.toolbox.merge_batch_bellman_operators(k, i, policy_index, j)

                    # Eliminate placeholders at session (k, i, j)
                    os.remove(
                        f"offline_data/bellman_operators/{k}/{i}/{policy_index}/{j}/{self.placeholder}"
                    )


if __name__ == "__main__":
    # Prepare RolloutEvaluator and offline data
    evaluator = RolloutEvaluator(
        gamma=parser.args.gamma,
        max_horizon=parser.args.rollout_max_horizon,
    )
    evaluator.start_rollout_on_single_session()
