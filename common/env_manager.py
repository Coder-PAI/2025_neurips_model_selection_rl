import copy
import os
import parser
from typing import Dict, Union

import gym
import numpy as np
from gym.envs.mujoco import MujocoEnv
from numpy import array

from global_utils import GeneralUtils


class MDPManager:
    def __init__(self, verbose):
        """
        Set verbose=1 to print more information.
        """
        self.verbose = verbose
        self.toolbox = GeneralUtils()

    def manage(self, args=parser.args):
        param_names = [
            "hopper_gravities",
            "hopper_noise_mean_list",
            "hopper_noise_scaler_list",
        ]
        parameters_list = [
            {
                "gravity": np.array([0.0, 0.0, gravity]),
                "force_mean": force_mean,
                "force_scaler": force_scaler,
            }
            for gravity, force_mean, force_scaler in [
                list(row) for row in zip(*[getattr(args, name) for name in param_names])
            ]
        ]

        mdp_class = self.make_default(mode="human", parameters_list=parameters_list)
        self.save_offline_table(mdp_class)

    def make_default(self, mode, parameters_list):
        names = [[str(parameters)] for parameters in parameters_list]
        mdp_class = zip(
            names,
            [
                self.toolbox.hopper_create_with_params(params, mode)
                for params in parameters_list
            ],
        )
        return mdp_class

    def save_offline_table(self, mdp_class):
        os.makedirs("offline_data/table_mdp", exist_ok=True)
        self.toolbox.save_as_pkl(
            "offline_data/table_mdp/index_to_name_and_params",
            {
                index: (
                    mdp_name,
                    self.toolbox.hopper_get_params(mdp_env),
                )
                for index, (mdp_name, mdp_env) in enumerate(mdp_class)
            },
        )
        self.toolbox.save_as_pkl(
            "offline_data/table_mdp/mdp_name_to_index",
            {mdp_name: index for index, (mdp_name, mdp_env) in enumerate(mdp_class)},
        )
