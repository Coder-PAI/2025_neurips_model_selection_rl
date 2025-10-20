import os
import parser
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from global_utils import GeneralUtils, StochasticEnv, device


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


def rollout_single_episode(args):
    toolbox, eval_params, policy_path, gamma = args
    eval_env = toolbox.hopper_create_with_params(eval_params)
    policy = torch.load(policy_path)
    if eval_env is not None:
        obs, _ = eval_env.reset()
        done = False
        episode_return = 0
        h = 0

        while not done:
            obs = torch.tensor(obs, device=device).unsqueeze(dim=0)
            action = policy(obs)[0].detach().cpu().numpy()
            obs, reward, done, _, _ = eval_env.step(action)
            episode_return += gamma**h * reward
            h += 1
        return episode_return
    return None


class CustomKVWriter(KVWriter):
    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Tuple[str, ...]],
        step: int = 0,
    ) -> None:
        super().write(key_values, key_excluded, step)
        for key, value in key_values.items():
            if key.startswith("custom/"):
                print(f"{key}: {value}")
            else:
                continue


class IndividualCheckpointCallback(BaseCallback):
    def __init__(self, eval_env, gamma, ckpt_cycle, save_path, verbose=True):
        super().__init__(verbose)
        self.ckpt_cycle = ckpt_cycle
        self.save_path = save_path
        self.eval_env = eval_env
        self.gamma = gamma
        self.toolbox = GeneralUtils()
        self.writer = SummaryWriter(os.path.join(save_path, "training_logs"))
        self.latest_mean_ep = None

    def rollout_single_episode(self):
        if self.eval_env is not None:
            obs, _ = self.eval_env.reset()
            done = False
            episode_return = 0
            h = 0

            while not done:
                obs = torch.tensor(obs, device=device).unsqueeze(dim=0)
                action = self.model.policy(obs)[0].detach().cpu().numpy()
                obs, reward, done, _, _ = self.eval_env.step(action)
                episode_return += self.gamma**h * reward
                h += 1
            return episode_return
        return None

    def rollout_eval(self, num_episodes=32):
        eval_params = self.toolbox.hopper_get_params(self.eval_env)
        os.makedirs(f"offline_data/policies/cache/", exist_ok=True)
        policy_cache_path = f"offline_data/policies/cache/policy.pth"
        torch.save(self.model.policy, policy_cache_path)
        multi_args = [
            (self.toolbox, eval_params, policy_cache_path, self.gamma)
            for _ in range(num_episodes)
        ]
        num_cores = available_cpu_cores(percent=10, limit=28)
        with Pool(processes=num_cores) as pool:
            episode_returns = list(
                tqdm(
                    pool.imap(rollout_single_episode, multi_args),
                    total=len(multi_args),
                    desc="Estimating episode return...",
                )
            )
        mean_ep = np.mean(episode_returns)
        self.latest_mean_ep = mean_ep
        self.logger.record("custom/ep_latest", mean_ep)
        self.logger.dump(0)
        os.remove(policy_cache_path)
        return mean_ep

    def env_adjust(self):
        self.model.learning_rate *= 1.01
        self.model._setup_lr_schedule()

    def _on_step(self) -> bool:
        if self.n_calls % self.ckpt_cycle == 0:
            os.makedirs(f"offline_data/policies/training_logs/", exist_ok=True)
            self.writer.add_scalar(
                f"offline_data/policies/training_logs/eval",
                self.rollout_eval(),
                self.num_timesteps,
            )
            index = self.n_calls // self.ckpt_cycle - 1
            ckpt_path = os.path.join(self.save_path, f"{index}")
            os.makedirs(ckpt_path, exist_ok=True)
            model_save_path = os.path.join(ckpt_path, f"DDPG.pth")
            torch.save(self.model.policy, model_save_path)
            self.env_adjust()
            if self.verbose:
                self.toolbox.render_text(
                    f"Policies {index} finished.",
                    "YELLOW",
                )
        return True


class PolicyTrainer:
    def __init__(
        self,
        index_to_name_and_params: Dict[int, Tuple[str, StochasticEnv]],
        device: str,
        epsilon: float,
        total_steps: int,
        checkpoint: int,
        lr: float,
        hidden_layers: List[int],
        limit: int,
        algorithm: str,
        verbose=True,
        clear_ckpt_after_train=True,
    ):
        self.index_to_name_and_params = index_to_name_and_params
        self.epsilon = epsilon
        self.total_steps = total_steps
        self.checkpoint = checkpoint
        self.lr = lr
        self.limit = limit
        self.hidden_layers = hidden_layers
        self.algorithm = algorithm
        self.device = device
        self.verbose = verbose
        self.clear_ckpt_after_train = clear_ckpt_after_train
        self.toolbox = GeneralUtils()

    def train_from_checkpoints(self):
        total_count = len(parser.args.hopper_gravities)
        ckpt_cycle = self.total_steps // total_count

        if all(
            [
                os.path.exists(f"offline_data/policies/{index}/{self.algorithm}.pth")
                for index in range(total_count)
            ]
        ):
            for index in range(total_count):
                self.toolbox.render_text(
                    f"\tPolicy {index} already exists. Checkpoints cleared out...",
                    color="RED",
                )
            return
        # Reach the corresponding MDP
        params = {
            "gravity": [0, 0, -30],
            "force_mean": 0,
            "force_scaler": 32,
        }
        env_train = self.toolbox.hopper_create_with_params(params)
        env_eval = self.toolbox.hopper_create_with_params(params)
        # Training
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh, net_arch=parser.args.hidden_layers
        )
        checkpoint_callback = IndividualCheckpointCallback(
            eval_env=env_eval,
            gamma=parser.args.gamma,
            ckpt_cycle=ckpt_cycle,
            save_path=f"offline_data/policies/",
        )
        model = DDPG(
            "MlpPolicy",
            env_train,
            verbose=1,
            tensorboard_log=f"offline_data/policies/training_logs",
            device=device,
            learning_rate=self.lr,
            policy_kwargs=policy_kwargs,
        )
        model.learn(
            total_timesteps=self.total_steps,
            callback=checkpoint_callback,
            log_interval=999999999999,
        )
