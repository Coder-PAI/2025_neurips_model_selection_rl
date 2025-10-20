import copy
import math
import os.path
import parser
import random
import re
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from global_utils import GeneralUtils, StochasticEnv


class SelectionBaselines:
    """
    Each selection algorithm returns a list of indices as the environments ranked by different priorities.

    Ideally, only the given policy, the dataset as well as a class of candidate environments should be exposed to each
    selection algorithm, and directly passing dataset_index seems to have leaked the answers. The reason why we would
    use corresponding indices is that we may need to load pre-cached offline Q/V functions & bellman operators in some
    algorithms which would require the exact index for the environment and policy, or we have to repeat simulations
    over and over again which takes excessively long time to complete the whole selection process (as the algorithm
    itself cannot reach pre-cached Q/V functions & bellman operators if only the policy, dataset and env_class is given
    without being indexed). And it is guaranteed that, in our implementations of each algorithm, the selectors can
    never cheat. Readers can give them a thorough check as they like.
    """

    # noinspection SpellCheckingInspection
    def __init__(self, method="one_pass", sample_range=None):
        self.utils = SelectionUtils()
        self.toolbox = GeneralUtils()
        self.baselines = [
            # output random results
            "trivial_random",
            # choose Q s.t. td square is minimized
            "model_free_td_square",
            # choose Q s.t. average bellman error is minimized
            "model_free_avg_bellman_error",
            # lstdq-inspired methods
            "model_free_lstdq_no_minus",
            "model_free_lstdq_minus",
            # bvft tournaments
            "model_free_bvft",
            # choose Q s.t. ||Q-TQ|| is minimized, where T is the estimated bellman operator for the ground truth
            "model_based_sbv",
            # absolute bellman loss inspired methods
            "model_based_sign_flip",
            # directly measure the loss for predicted states
            "model_based_naive",
            "comparison_bvft_disc",
            "comparison_lstdq_oracle",
        ]
        if not parser.args.enable_bellman_operators:
            self.baselines = [
                alg
                for alg in self.baselines
                if "model_based" not in alg or alg == "model_based_naive"
            ]
        if method == "one_pass":
            self.discriminators = DiscriminatorsWithDirectInput()
        elif method == "incremental":
            self.discriminators = DifferentSizeDiscriminatorsWithDirectInput(
                sample_range=sample_range
            )
        else:
            raise AssertionError(
                f"Method {method} not supported! Expected to be one of the following: "
                f"[one_pass, incremental]."
            )

    # noinspection SpellCheckingInspection
    def __getattr__(self, algorithm):
        """
        __getattr__ method designed exclusively for BVFT methods with specified resolutions.
        For example, when you would like to call BVFT with a resolution r=12,
        you can use the API as SelectionBaselines().model_free_bvft_12.
        """
        pattern = r"^model_free_bvft_(\d+(\.\d+)?)$"
        match = re.match(pattern, algorithm)
        if match:
            resolution = float(match.group(1))
            self.toolbox.render_text(
                f"Using BVFT discriminators of resolution {resolution}.", "RED"
            )
            return partial(self.model_free_bvft, resolutions=[resolution])
        else:
            return False

    @staticmethod
    def resample(
        bellman_operators, q_cur_tables, v_next_tables, rewards, size, specify_indices
    ):
        # check alignment
        assert len(q_cur_tables) == len(v_next_tables)
        if specify_indices is not None:
            assert len(set(specify_indices)) == len(specify_indices) == size
        for q_cur_table, v_next_table in zip(q_cur_tables, v_next_tables):
            assert len(q_cur_table) == len(v_next_table) == len(rewards)
        if bellman_operators is not None:
            for t_tables in bellman_operators:
                for t_table in t_tables:
                    assert len(t_table) == len(rewards)

        # return original data if resampling is not required
        if size is None:
            return bellman_operators, q_cur_tables, v_next_tables, rewards, None

        # resample data with replacement
        total_size = len(rewards)
        if specify_indices is None:
            resampled_indices = np.random.choice(range(total_size), size, replace=True)
        else:
            resampled_indices = specify_indices

        def subset(data, indices):
            return [data[i] for i in indices]

        resampled_q = [subset(q_table, resampled_indices) for q_table in q_cur_tables]
        resampled_v = [subset(v_table, resampled_indices) for v_table in v_next_tables]
        resampled_r = subset(rewards, resampled_indices)
        if bellman_operators is not None:
            resampled_t = [
                [subset(t_table, resampled_indices) for t_table in t_tables]
                for t_tables in bellman_operators
            ]
        else:
            resampled_t = bellman_operators

        return resampled_t, resampled_q, resampled_v, resampled_r, resampled_indices

    def preparation(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size,
        require_bellman_operators,
        specify_indices,
        return_indices=False,
    ):
        # Preparations
        q_cur_tables, v_next_tables, dataset = self.toolbox.preparations(
            policy_index=policy_index,
            env_indices=env_indices,
            dataset_index=dataset_index,
        )
        all_rewards = list(zip(*self.toolbox.all_data_from_dataset(dataset)))[4]

        if require_bellman_operators:
            all_env_indices = range(len(parser.args.hopper_gravities))
            bellman_operators = []
            for env_index in all_env_indices:
                bellman_operators.append([])
                for k in all_env_indices:
                    raw_bellman_operators = self.toolbox.load_bellman_operator(
                        env_cand_index=k,
                        env_q_index=env_index,
                        policy_index=policy_index,
                        dataset_index=dataset_index,
                    )
                    flattened_bellman_operators = [
                        bellman_op_processed_q
                        for trajectory in raw_bellman_operators
                        for bellman_op_processed_q in trajectory
                    ]
                    assert len(flattened_bellman_operators) == len(all_rewards)
                    bellman_operators[-1].append(flattened_bellman_operators)
        else:
            bellman_operators = None

        # resample
        (
            bellman_operators,
            q_cur_tables,
            v_next_tables,
            all_rewards,
            indices,
        ) = self.resample(
            bellman_operators,
            q_cur_tables,
            v_next_tables,
            all_rewards,
            sample_size,
            specify_indices,
        )

        if require_bellman_operators:
            if return_indices:
                return (
                    np.array(bellman_operators),
                    q_cur_tables,
                    v_next_tables,
                    all_rewards,
                    indices,
                )
            else:
                return (
                    np.array(bellman_operators),
                    q_cur_tables,
                    v_next_tables,
                    all_rewards,
                )
        else:
            if return_indices:
                return None, q_cur_tables, v_next_tables, all_rewards, indices
            else:
                return (
                    None,
                    q_cur_tables,
                    v_next_tables,
                    all_rewards,
                )

    def trivial_random(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        data_size = len(all_rewards)
        return self.discriminators.trivial_random(env_indices=env_indices)

    def model_free_td_square(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_td_square(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    def model_free_avg_bellman_error(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_avg_bellman_error(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    # noinspection SpellCheckingInspection
    def model_free_lstdq_no_minus(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_lstdq_no_minus(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    # noinspection SpellCheckingInspection
    def model_free_lstdq_minus(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_lstdq_minus(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    # noinspection SpellCheckingInspection
    def model_free_hybrid_lstdq(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_hybrid_lstdq(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    # noinspection SpellCheckingInspection
    def model_free_std_norm_lstdq_minus(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_std_norm_lstdq_minus(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    # noinspection SpellCheckingInspection
    def model_free_std_norm_lstdq_no_minus(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_std_norm_lstdq_no_minus(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    # noinspection SpellCheckingInspection
    def model_free_bvft(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
        enable_plot=False,
        resolutions=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_free_bvft(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
            enable_plot=enable_plot,
            resolutions=resolutions,
        )

    def model_based_sbv(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        bellman_operators, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=True,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_based_sbv(
            env_indices=env_indices,
            bellman_operators=bellman_operators,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    def model_based_fitted_bellman_square(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        bellman_operators, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=True,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_based_fitted_bellman_square(
            env_indices=env_indices,
            bellman_operators=bellman_operators,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    def model_based_sign_flip(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        bellman_operators, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=True,
            specify_indices=specify_indices,
        )
        return self.discriminators.model_based_sign_flip(
            env_indices=env_indices,
            bellman_operators=bellman_operators,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    def model_based_naive(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        dataset = self.toolbox.load_dataset(dataset_index)
        if isinstance(self.discriminators, DifferentSizeDiscriminatorsWithDirectInput):
            return self.discriminators.model_based_naive(
                env_indices=env_indices,
                dataset=dataset,
            )
        elif isinstance(self.discriminators, DiscriminatorsWithDirectInput):
            if specify_indices is None:
                data_size = sum(episode["num_states"] - 1 for episode in dataset)
                assert data_size >= parser.args.sampler_size
                specify_indices = list(range(data_size))
            return self.discriminators.model_based_naive(
                env_indices=env_indices,
                dataset=dataset,
                resample_indices=specify_indices,
            )

    # noinspection SpellCheckingInspection
    def comparison_lstdq_oracle(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
    ):
        _, q_cur_tables, v_next_tables, all_rewards, indices = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            return_indices=True,
            specify_indices=specify_indices,
        )
        ground_q = [
            q
            for trajectory in self.toolbox.load_q_function(
                env_index=dataset_index,
                policy_index=policy_index,
                dataset_index=dataset_index,
            )
            for q in trajectory
        ]
        if indices is not None:
            ground_q = [ground_q[index] for index in indices]

        return self.discriminators.comparison_lstdq_oracle(
            ground_q=ground_q,
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
        )

    # noinspection SpellCheckingInspection
    def comparison_bvft_disc(
        self,
        policy_index,
        env_indices,
        dataset_index,
        sample_size=None,
        specify_indices=None,
        enable_plot=False,
    ):
        _, q_cur_tables, v_next_tables, all_rewards = self.preparation(
            policy_index,
            env_indices,
            dataset_index,
            sample_size,
            require_bellman_operators=False,
            specify_indices=specify_indices,
        )
        return self.discriminators.comparison_bvft_disc(
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
            enable_plot=enable_plot,
        )


class DiscriminatorsWithDirectInput:
    def __init__(self):
        self.toolbox = GeneralUtils()
        self.utils = SelectionUtils()

    # noinspection SpellCheckingInspection
    @staticmethod
    def model_free_lstdq_no_minus(
        env_indices, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        loss_over_all_envs = []
        utils = LSTDQUtils()
        for q1_list, v_list in zip(q_cur_tables, v_next_tables):
            r_plus_v = [
                reward + parser.args.gamma * v for reward, v in zip(all_rewards, v_list)
            ]
            loss = max(
                math.fabs(utils.no_minus_loss(q1_list, q2_list, r_plus_v))
                for q2_list in q_cur_tables
            )
            loss_over_all_envs.append(loss)
        # print(loss_over_all_envs)
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    # noinspection SpellCheckingInspection
    @staticmethod
    def model_free_lstdq_minus(
        env_indices, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        loss_over_all_envs = []
        utils = LSTDQUtils()
        for q1_list, v_list in zip(q_cur_tables, v_next_tables):
            r_plus_v = [
                reward + parser.args.gamma * v for reward, v in zip(all_rewards, v_list)
            ]
            loss = max(
                math.fabs(utils.minus_loss(q1_list, q2_list, r_plus_v))
                for q2_list in q_cur_tables
            )
            loss_over_all_envs.append(loss)
        # print(loss_over_all_envs)
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    # noinspection SpellCheckingInspection
    @staticmethod
    def model_free_hybrid_lstdq(
        env_indices, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        loss_over_all_envs = []
        utils = LSTDQUtils()
        for i, (q1_list, v_list) in enumerate(zip(q_cur_tables, v_next_tables)):
            r_plus_v = [
                reward + parser.args.gamma * v for reward, v in zip(all_rewards, v_list)
            ]
            loss = max(
                math.fabs(utils.std_norm_minus_loss(q1_list, q2_list, r_plus_v))
                if i != j
                else math.fabs(utils.std_norm_no_minus_loss(q1_list, q2_list, r_plus_v))
                for j, q2_list in enumerate(q_cur_tables)
            )
            loss_over_all_envs.append(loss)
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    # noinspection SpellCheckingInspection
    @staticmethod
    def model_free_std_norm_lstdq_minus(
        env_indices, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        loss_over_all_envs = []
        utils = LSTDQUtils()
        for i, (q1_list, v_list) in enumerate(zip(q_cur_tables, v_next_tables)):
            r_plus_v = [
                reward + parser.args.gamma * v for reward, v in zip(all_rewards, v_list)
            ]
            loss = max(
                math.fabs(utils.std_norm_minus_loss(q1_list, q2_list, r_plus_v))
                for j, q2_list in enumerate(q_cur_tables)
            )
            loss_over_all_envs.append(loss)
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    # noinspection SpellCheckingInspection
    @staticmethod
    def model_free_std_norm_lstdq_no_minus(
        env_indices, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        loss_over_all_envs = []
        utils = LSTDQUtils()
        for i, (q1_list, v_list) in enumerate(zip(q_cur_tables, v_next_tables)):
            r_plus_v = [
                reward + parser.args.gamma * v for reward, v in zip(all_rewards, v_list)
            ]
            loss = max(
                math.fabs(utils.std_norm_no_minus_loss(q1_list, q2_list, r_plus_v))
                for j, q2_list in enumerate(q_cur_tables)
            )
            loss_over_all_envs.append(loss)
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    # noinspection SpellCheckingInspection
    @staticmethod
    def comparison_lstdq_oracle(
        env_indices, ground_q, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        loss_over_all_envs = []
        utils = LSTDQUtils()
        for q1_list, v_list in zip(q_cur_tables, v_next_tables):
            r_plus_v = [
                reward + parser.args.gamma * v for reward, v in zip(all_rewards, v_list)
            ]
            loss = math.fabs(utils.no_minus_loss(q1_list, ground_q, r_plus_v))
            loss_over_all_envs.append(loss)
        # print(loss_over_all_envs)
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    # noinspection SpellCheckingInspection
    def model_free_bvft(
        self,
        env_indices,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        enable_plot=False,
        resolutions=None,
        **kwargs,
    ):
        return self.bvft_selection(
            disc_comparison=False,
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
            enable_plot=enable_plot,
            resolutions=resolutions,
            **kwargs,
        )

    # noinspection SpellCheckingInspection
    def comparison_bvft_disc(
        self,
        env_indices,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        enable_plot=False,
        resolutions=None,
        **kwargs,
    ):
        return self.bvft_selection(
            disc_comparison=True,
            env_indices=env_indices,
            q_cur_tables=q_cur_tables,
            v_next_tables=v_next_tables,
            all_rewards=all_rewards,
            enable_plot=enable_plot,
            resolutions=resolutions,
            **kwargs,
        )

    # noinspection SpellCheckingInspection
    def bvft_selection(
        self,
        disc_comparison,
        env_indices,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        enable_plot=False,
        resolutions=None,
        **kwargs,
    ):
        # Create resolution grids
        loss_over_all_resolutions = []
        ranking_over_all_resolutions = []
        # create legal resolutions
        if resolutions is None:
            # fetch legal resolution range
            res_ranges = [max(q_table) - min(q_table) for q_table in q_cur_tables]
            max_resolution = min(res_ranges) * 0.125
            resolutions = list(np.linspace(0.1, max_resolution, 10))
        for r in resolutions:
            utils = BVFTUtils(r)
            # losses over all the environments & the ranking at resolution r
            loss_over_all_envs = utils.one_pass_given_resolution(
                env_indices,
                q_cur_tables,
                v_next_tables,
                all_rewards,
                disc_comparison=disc_comparison,
            )
            ranking = [env_indices[i] for i in np.argsort(loss_over_all_envs)]
            # record loss(r) and ranking(r) into lists
            loss_over_all_resolutions.append(copy.deepcopy(loss_over_all_envs))
            ranking_over_all_resolutions.append(copy.deepcopy(ranking))
            if r == resolutions[-1] and enable_plot:
                utils.save_diff_resolutions(
                    resolutions, [min(loss) for loss in loss_over_all_resolutions]
                )
                utils.plot_diff_resolutions()
        opt_index = np.argmin([min(loss) for loss in loss_over_all_resolutions])
        return (
            ranking_over_all_resolutions[opt_index],
            loss_over_all_resolutions[opt_index],
        )

    @staticmethod
    def model_based_sign_flip(
        env_indices,
        bellman_operators,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        **kwargs,
    ):
        utils = AbsIndicatorUtils(env_indices)
        loss_over_all_envs = [
            utils.abs(env_index, bellman_operators, all_rewards, q_list, v_list)
            for env_index, q_list, v_list in zip(
                env_indices, q_cur_tables, v_next_tables
            )
        ]
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    def model_based_naive(self, env_indices, dataset, resample_indices, **kwargs):
        env_class = [self.toolbox.load_env(env_index) for env_index in env_indices]
        utils = NaiveModelBasedUtils(dataset)
        loss_over_all_envs = [
            utils.naive_loss(env, resample_indices) for env in env_class
        ]
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    @staticmethod
    def model_based_sbv(
        env_indices,
        bellman_operators,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        **kwargs,
    ):
        """
        Assume the dataset is indexed as i and the policy is indexed as j (where i and j together conceptually remain
        unknown to the algorithm), then the target Q function is the solution to the following equation:
        Q(s, a) - T_{M_i}^{pi_j} Q(s, a) = 0 (for any (s, a) that belongs to D_i).
        However, as we do not intellectually know anything about T_{M_i}^{pi_j}:=T^*, our first step is to fit T^* by
        the given environment class which is currently known to us; we are then able to rank the environments according
        to the loss ||Q-hat{T}Q|| where hat{T} is the fitted estimation for T^*.
        """
        loss_over_all_envs = []
        utils = BellmanBackupUtils(env_indices)
        for i, env_index in enumerate(env_indices):
            # Find hat{T}
            env_cand_index = utils.fit(
                env_index, bellman_operators, v_next_tables[i], all_rewards
            )
            # Compute ||Q-hat{T}Q||
            loss_over_all_envs.append(
                utils.loss(
                    env_cand_index, env_index, bellman_operators, q_cur_tables[i]
                )
            )
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    def model_based_fitted_bellman_square(
        self,
        env_indices,
        bellman_operators,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        **kwargs,
    ):
        """
        Assume the dataset is indexed as i and the policy is indexed as j (where i and j together conceptually remain
        unknown to the algorithm), then the target Q function is the solution to the following equation:
        Q(s, a) - T_{M_i}^{pi_j} Q(s, a) = 0 (for any (s, a) that belongs to D_i).
        However, as we do not intellectually know anything about T_{M_i}^{pi_j}:=T^*, our first step is to fit T^* by
        the given environment class which is currently known to us; we are then able to rank the environments according
        to the loss ||Q-hat{T}Q|| where hat{T} is the fitted estimation for T^*.
        """
        td_square_terms = self.model_free_td_square(
            env_indices, q_cur_tables, v_next_tables, all_rewards
        )[1]
        fitted_terms = []
        fitted_utils = BellmanBackupUtils(env_indices)
        td_utils = SelectionUtils()

        for i, env_index in enumerate(env_indices):
            # Find hat{T}
            env_cand_index = fitted_utils.fit(
                env_index, bellman_operators, v_next_tables[i], all_rewards
            )
            q_list = bellman_operators[env_index][env_cand_index]
            fitted_loss = td_utils.td_square_over_dataset(
                q_cur_list=q_list, v_next_list=v_next_tables[i], rewards=all_rewards
            )
            fitted_terms.append(fitted_loss)
        assert len(fitted_terms) == len(td_square_terms)
        loss_over_all_envs = [
            (td - fit) for td, fit in zip(td_square_terms, fitted_terms)
        ]
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    @staticmethod
    def trivial_random(**kwargs):
        env_indices = kwargs["env_indices"]
        env_indices = sorted(env_indices, key=lambda *args: random.random())
        return env_indices, None

    @staticmethod
    def trivial_ascending(**kwargs):
        env_indices = kwargs["env_indices"]
        return sorted(env_indices), None

    @staticmethod
    def trivial_descending(**kwargs):
        env_indices = kwargs["env_indices"]
        return sorted(env_indices, reverse=True), None

    @staticmethod
    def model_free_td_square(
        env_indices, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        utils = SelectionUtils()
        loss_over_all_envs = [
            utils.td_square_over_dataset(q_cur_list, v_next_list, all_rewards)
            for q_cur_list, v_next_list in zip(q_cur_tables, v_next_tables)
        ]
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs

    def model_free_avg_bellman_error(
        self, env_indices, q_cur_tables, v_next_tables, all_rewards, **kwargs
    ):
        loss_over_all_envs = [
            math.fabs(self.utils.bellman_error_over_dataset(q, v, all_rewards))
            for q, v in zip(q_cur_tables, v_next_tables)
        ]
        # print(loss_over_all_envs)
        return [
            env_indices[i] for i in np.argsort(loss_over_all_envs)
        ], loss_over_all_envs


class DifferentSizeDiscriminatorsWithDirectInput:
    def __init__(self, sample_range):
        self.sample_range = sample_range
        self.discriminator = DiscriminatorsWithDirectInput()

    def resample_indices(self, all_sample_indices):
        for sample_size in self.sample_range:
            yield all_sample_indices[:sample_size]

    def resample_function_estimates(self, functions_table, all_sample_indices):
        array_table = np.array(functions_table)
        assert len(array_table.shape) == 2
        return [
            array_table[:, indices].tolist()
            for indices in self.resample_indices(all_sample_indices)
        ]

    def resample_bellman_operators(self, bellman_operators, all_sample_indices):
        array_t = np.array(bellman_operators)
        assert len(array_t.shape) == 3
        return [
            array_t[:, :, indices].tolist()
            for indices in self.resample_indices(all_sample_indices)
        ]

    def resample_rewards(self, rewards, all_sample_indices):
        array_rewards = np.array(rewards)
        assert len(array_rewards.shape) == 1
        return [
            array_rewards[indices].tolist()
            for indices in self.resample_indices(all_sample_indices)
        ]

    def general_model_free_api(
        self,
        model_free_alg,
        env_indices,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        *args,
        **kwargs,
    ):
        total_size = parser.args.sampler_size
        all_sample_indices = list(
            np.random.choice(range(total_size), total_size, replace=True)
        )
        resampled_q_tables = self.resample_function_estimates(
            q_cur_tables, all_sample_indices
        )
        resampled_v_tables = self.resample_function_estimates(
            v_next_tables, all_sample_indices
        )
        resampled_rewards = self.resample_rewards(all_rewards, all_sample_indices)
        for q, v, r in zip(resampled_q_tables, resampled_v_tables, resampled_rewards):
            yield getattr(self.discriminator, model_free_alg)(
                env_indices=env_indices,
                q_cur_tables=q,
                v_next_tables=v,
                all_rewards=r,
                *args,
                **kwargs,
            )

    def general_model_based_api(
        self,
        model_based_alg,
        env_indices,
        bellman_operators,
        q_cur_tables,
        v_next_tables,
        all_rewards,
        *args,
        **kwargs,
    ):
        total_size = parser.args.sampler_size
        all_sample_indices = list(
            np.random.choice(range(total_size), total_size, replace=True)
        )
        resampled_q_tables = self.resample_function_estimates(
            q_cur_tables, all_sample_indices
        )
        resampled_v_tables = self.resample_function_estimates(
            v_next_tables, all_sample_indices
        )
        resampled_rewards = self.resample_rewards(all_rewards, all_sample_indices)
        resampled_t = self.resample_bellman_operators(
            bellman_operators, all_sample_indices
        )
        for t, q, v, r in zip(
            resampled_t, resampled_q_tables, resampled_v_tables, resampled_rewards
        ):
            yield getattr(self.discriminator, model_based_alg)(
                env_indices=env_indices,
                bellman_operators=t,
                q_cur_tables=q,
                v_next_tables=v,
                all_rewards=r,
                *args,
                **kwargs,
            )

    def model_based_naive(self, env_indices, dataset):
        total_size = parser.args.sampler_size
        all_sample_indices = list(
            np.random.choice(range(total_size), total_size, replace=True)
        )
        for sample_indices in self.resample_indices(all_sample_indices):
            yield self.discriminator.model_based_naive(
                env_indices=env_indices,
                dataset=dataset,
                resample_indices=sample_indices,
            )

    def trivial_random(self, env_indices):
        for _ in self.sample_range:
            yield self.discriminator.trivial_random(env_indices=env_indices)

    def __getattr__(self, alg):
        if "model_free" in alg:
            return partial(self.general_model_free_api, model_free_alg=alg)
        elif "model_based" in alg:
            return partial(self.general_model_based_api, model_based_alg=alg)
        else:
            raise AttributeError(
                "Algorithm must include model_free or model_based labels!"
            )


class SelectionUtils:
    def __init__(self):
        self.toolbox = GeneralUtils()

    @staticmethod
    def bellman_error_for_single_sample(q_cur: float, v_next: float, reward):
        return q_cur - reward - parser.args.gamma * v_next

    def bellman_error_over_dataset(
        self, qs_cur: List[float], vs_next: List[float], rewards
    ):
        assert len(qs_cur) == len(vs_next) == len(rewards)
        return np.sum(
            [
                self.bellman_error_for_single_sample(q_cur, v_next, reward)
                for q_cur, v_next, reward in zip(qs_cur, vs_next, rewards)
            ]
        )

    @staticmethod
    def td_square_for_single_sample(q_cur, v_next, reward):
        return (q_cur - reward - parser.args.gamma * v_next) ** 2

    def td_square_over_dataset(self, q_cur_list, v_next_list, rewards):
        assert len(q_cur_list) == len(v_next_list) == len(rewards)
        return np.mean(
            [
                self.td_square_for_single_sample(q_cur, v_next, reward)
                for q_cur, v_next, reward in zip(q_cur_list, v_next_list, rewards)
            ]
        )


class NaiveModelBasedUtils(SelectionUtils):
    def __init__(self, dataset, state_norm=2):
        super().__init__()
        self.dataset = dataset
        self.q_pos_list = [
            q_pos for episode in self.dataset for q_pos in episode["q_pos"][:-1]
        ]
        self.q_vel_list = [
            q_vel for episode in self.dataset for q_vel in episode["q_vel"][:-1]
        ]
        self.action_list = [
            action for episode in self.dataset for action in episode["actions"]
        ]
        self.state_list = [
            state for episode in self.dataset for state in episode["observations"][:-1]
        ]
        self.next_state_list = [
            state for episode in self.dataset for state in episode["observations"][1:]
        ]
        assert (
            len(self.q_pos_list)
            == len(self.q_vel_list)
            == len(self.state_list)
            == len(self.next_state_list)
            == len(self.action_list)
        )
        self.state_norm = state_norm

    def state_loss_for_single_pair(self, env, q_pos, q_vel, action, next_state):
        env.reset()
        env.set_state(q_pos, q_vel)

        next_obs, _, _, _, _ = env.step(action)
        return np.linalg.norm(next_obs - next_state, ord=self.state_norm)

    def naive_loss(self, env: StochasticEnv, resample_indices):
        return np.mean(
            [
                self.state_loss_for_single_pair(
                    env,
                    self.q_pos_list[index],
                    self.q_vel_list[index],
                    self.action_list[index],
                    self.next_state_list[index],
                )
                for index in resample_indices
            ]
        )


class AbsIndicatorUtils(SelectionUtils):
    def __init__(self, env_indices):
        super().__init__()
        self.env_indices = env_indices

    def cheat(
        self, ground_index, env_index, bellman_operators, all_rewards, q_list, v_list
    ):
        assert len(q_list) == len(v_list) == len(all_rewards)
        for t_table in bellman_operators[env_index]:
            assert len(t_table) == len(all_rewards)
        bellman_op_over_envs = np.array(bellman_operators[env_index])
        bellman_errors_without_indicator = np.array(
            [
                self.bellman_error_for_single_sample(q_cur, v_next, reward)
                for q_cur, v_next, reward in zip(q_list, v_list, all_rewards)
            ]
        )
        indicators = np.sign(
            [q - bellman_op_over_envs[ground_index][k] for k, q in enumerate(q_list)]
        )
        bellman_errors_with_indicator = bellman_errors_without_indicator * indicators
        return np.mean(bellman_errors_with_indicator)

    def abs(self, env_index, bellman_operators, all_rewards, q_list, v_list):
        assert len(q_list) == len(v_list) == len(all_rewards)
        for t_table in bellman_operators[env_index]:
            assert len(t_table) == len(all_rewards)
        bellman_op_over_envs = np.array(bellman_operators[env_index])
        # bellman_errors_without_indicator is shaped (|D|, )
        bellman_errors_without_indicator = np.array(
            [
                self.bellman_error_for_single_sample(q_cur, v_next, reward)
                for q_cur, v_next, reward in zip(q_list, v_list, all_rewards)
            ]
        )
        # the array "indicators" is shaped (L, |D|)
        indicators = np.sign(
            [
                [q - bellman_op_over_envs[h][k] for k, q in enumerate(q_list)]
                for h in self.env_indices
            ]
        )
        # bellman_errors_with_indicator is shaped (L, |D|)
        bellman_errors_with_indicator = (
            np.tile(
                bellman_errors_without_indicator,
                (len(self.env_indices), 1),
            )
            * indicators
        )
        # loss_under_all_operators is shaped (|L|, )
        loss_under_all_operators = (
            np.abs(np.sum(bellman_errors_with_indicator, axis=1))
            / bellman_errors_with_indicator.shape[1]
        )
        return max(loss_under_all_operators)


class LSTDQUtils(SelectionUtils):
    @staticmethod
    def no_minus_loss(q1_list, q2_list, r_plus_v):
        assert len(q1_list) == len(q2_list) == len(r_plus_v)
        return np.mean(
            [q2 * (q1 - tq) for q1, q2, tq in zip(q1_list, q2_list, r_plus_v)]
        )

    @staticmethod
    def minus_loss(q1_list, q2_list, r_plus_v):
        assert len(q1_list) == len(q2_list) == len(r_plus_v)
        return np.mean(
            [(q1 - q2) * (q1 - tq) for q1, q2, tq in zip(q1_list, q2_list, r_plus_v)]
        )

    @staticmethod
    def std_norm_no_minus_loss(q1_list, q2_list, r_plus_v):
        assert len(q1_list) == len(q2_list) == len(r_plus_v)
        std_q2 = np.std(q2_list) if len(q2_list) > 1 else 1
        return (
            np.mean([q2 * (q1 - tq) for q1, q2, tq in zip(q1_list, q2_list, r_plus_v)])
            / std_q2
        )

    @staticmethod
    def std_norm_minus_loss(q1_list, q2_list, r_plus_v):
        assert len(q1_list) == len(q2_list) == len(r_plus_v)
        std = (
            np.std([q1 - q2 for q1, q2 in zip(q1_list, q2_list)])
            if len(q1_list) > 1
            else 1
        )
        return (
            np.mean(
                [
                    (q1 - q2) * (q1 - tq)
                    for q1, q2, tq in zip(q1_list, q2_list, r_plus_v)
                ]
            )
            / std
        )


class BellmanBackupUtils(SelectionUtils):
    def __init__(self, env_indices):
        super().__init__()
        self.env_indices = env_indices

    def fit(self, env_index, bellman_operators, v_list, all_rewards):
        """
        The function outputs the fitted bellman operator T which minimizes ||r + gamma * Q - TQ||_2^2, where
        Q:=Q_{M_i}(pi_j) and i is the index of a given candidate environment.
        """
        loss_over_bellman_operators = []
        gamma = parser.args.gamma
        for i in self.env_indices:
            t_table = bellman_operators[env_index][i]
            assert len(t_table) == len(v_list) == len(all_rewards)
            loss = np.mean(
                [
                    (r + gamma * v - tq) ** 2
                    for r, v, tq in zip(all_rewards, v_list, t_table)
                ]
            )
            loss_over_bellman_operators.append(loss)
        # print(loss_over_bellman_operators)
        return self.env_indices[np.argmin(loss_over_bellman_operators)]

    @staticmethod
    def loss(env_cand_index, env_q_index, bellman_operators, q_list):
        """
        Once the fitted bellman operator T_{M_i} is determined with i as the env_cand_index, output the corresponding
        loss for a given candidate environment indexed by env_q_index.
        """
        bellman_operator = bellman_operators[env_q_index][env_cand_index]
        assert len(q_list) == len(bellman_operator)
        return np.mean([(q - tq) ** 2 for q, tq in zip(q_list, bellman_operator)])


class BVFTUtils(SelectionUtils):
    # noinspection SpellCheckingInspection
    def __init__(self, resolution):
        assert resolution > 0
        super().__init__()
        self.resolution = resolution
        self.colors = [
            "#FF0000",
            "#00FF00",
            "#800000",
            "#808000",
            "#0000FF",
            "#FFFF00",
            "#008000",
            "#800080",
            "#FF00FF",
            "#00FFFF",
        ]

    def shoot(self, x, q_grid):
        """
        The function is used to find the closest number in q_list to the given number x.
        """
        if x < q_grid[0]:
            return 0
        if x >= q_grid[-1]:
            return -1
        i = int((x - q_grid[0]) / self.resolution)
        cand_low, cand_up = q_grid[i], q_grid[(i + 1) % len(q_grid)]
        return (
            i
            if math.fabs(x - cand_low) <= math.fabs(cand_up - x)
            else (i + 1) % len(q_grid)
        )

    def discretize(self, flattened_q_list):
        """
        The function "discretize" takes a list containing different trajectories; each position in the single
        trajectory has a float number representing the corresponding q-estimation. The structure is almost the same as
        that for storing a single offline dataset.
        """
        # print(flattened_q_list[:10])
        max_q = np.max(flattened_q_list)
        min_q = np.min(flattened_q_list)
        q_grid = []
        q_pointer = min_q + self.resolution
        # print(min_q, max_q, q_pointer)
        # print(flattened_q_list)
        while q_pointer < max_q:
            q_grid.append(q_pointer)
            q_pointer += self.resolution
        indices = [self.shoot(q, q_grid) for q in flattened_q_list]
        flattened_q_disc = [q_grid[index] for index in indices]
        return flattened_q_disc, q_grid

    # noinspection SpellCheckingInspection
    @staticmethod
    def get_abstraction(flattend_q_list_1, flattend_q_list_2):
        """
        Given two discretized and flattened q lists, the function aims to return a list of groups where each
        index in the same group shares the same q1_disc and q2_disc simultaneously. For any indices i, j satisfying
        q1[i] = q1[j] and q2[i] = q2[j], the hash keys (q1, q2) for i and j should be equivalent to each other,
        based on which we can consider implementing the absraction by bins with keys (q1, q2).
        """
        q1_flattened = [q for q in flattend_q_list_1]
        q2_flattened = [q for q in flattend_q_list_2]
        bins = {(q1, q2): [] for q1, q2 in zip(q1_flattened, q2_flattened)}
        for index, (q1, q2) in enumerate(zip(q1_flattened, q2_flattened)):
            bins[(q1, q2)].append(index)
        return list(bins.values())

    @staticmethod
    def optimal_for_single_group(group_reward, group_v_next):
        assert len(group_reward) == len(group_v_next)
        return sum(
            reward + parser.args.gamma * v_next
            for reward, v_next in zip(group_reward, group_v_next)
        ) / len(group_reward)

    @staticmethod
    def get_group_data(group, flattened_rewards, flattened_v_next):
        group_reward = []
        group_v_next = []
        for index in group:
            group_reward.append(flattened_rewards[index])
            group_v_next.append(flattened_v_next[index])
        return group_reward, group_v_next

    # noinspection SpellCheckingInspection
    def bvft_operator(
        self, q_disc_list_1, q_disc_list_2, flattend_rewards, flattend_v_next
    ):
        """
        It is easy to see that once the abstraction (G_1, ..., G_h) is assigned, the BVFT loss can be decomposed into
        the sum of losses for different groups. To minimize the total loss, the function should ensure that the loss
        for each group is minimized independently (as can be inferred from the independence of the data in each group
        under the function g).

        The variable "flattened_v_next" is induced by "q_disc_list_1", which can be accessed by doing rollout
        estimations on the corresponding env and policy of "q_disc_list_1".
        """
        assert len(flattend_rewards) == len(flattend_v_next)
        abstraction = self.get_abstraction(q_disc_list_1, q_disc_list_2)
        g = [float("inf") for _ in range(len(flattend_rewards))]
        for group in abstraction:
            group_reward, group_v_next = self.get_group_data(
                group, flattend_rewards, flattend_v_next
            )
            optimal_g = self.optimal_for_single_group(group_reward, group_v_next)
            for index in group:
                g[index] = optimal_g
        return np.array(g)

    # noinspection SpellCheckingInspection
    @staticmethod
    def loss(bvft_operator, q_list_1):
        """
        Compute the loss "Epsilon(f1; f2)" as shown in the BVFT paper.
        """
        assert len(q_list_1) == len(bvft_operator)
        error_vector = np.array(q_list_1) - bvft_operator
        return np.linalg.norm(error_vector, ord=2) / math.sqrt(len(error_vector))

    # noinspection SpellCheckingInspection
    def initialization(self, q_cur_tables):
        # Discretization
        disc_q_lists = []
        q_grid_list = []
        for q_cur_list in q_cur_tables:
            assert len(q_cur_list) == len(q_cur_tables[0])
            disc_q_list, q_grid = self.discretize(q_cur_list)
            disc_q_lists.append(disc_q_list)
            q_grid_list.append(q_grid)
        return disc_q_lists, q_grid_list

    # noinspection SpellCheckingInspection
    def one_pass_given_resolution(
        self, env_indices, q_cur_tables, v_next_tables, all_rewards, disc_comparison
    ):
        # Discretization
        disc_q_lists, q_grid_list = self.initialization(q_cur_tables)

        # BVFT Tournament
        tournaments = [
            [float("inf") for _ in range(len(env_indices))]
            for _ in range(len(env_indices))
        ]
        assert len(disc_q_lists) == len(v_next_tables)
        for i, (q_list_1, disc_q_1, v_next_list) in enumerate(
            zip(q_cur_tables, disc_q_lists, v_next_tables)
        ):
            for j, disc_q_2 in enumerate(disc_q_lists):
                bvft_operator = self.bvft_operator(
                    disc_q_1, disc_q_2, all_rewards, v_next_list
                )
                if disc_comparison:
                    loss = self.loss(bvft_operator, disc_q_1)
                else:
                    loss = self.loss(bvft_operator, q_list_1)
                tournaments[i][j] = loss

        # Compute final loss
        loss_over_all_envs = [max(tournaments[i]) for i in range(len(env_indices))]

        return loss_over_all_envs

    # noinspection SpellCheckingInspection
    def save_diff_resolutions(self, resolutions, losses):
        count = 0
        os.makedirs(f"offline_data/bvft/", exist_ok=True)
        while os.path.exists(f"offline_data/bvft/{count}.pkl"):
            count += 1
        self.toolbox.save_as_pkl(f"offline_data/bvft/{count}", [resolutions, losses])

    # noinspection SpellCheckingInspection
    def plot_diff_resolutions(self):
        current_figure = plt.gcf()
        plt.figure()
        count = 0
        resolutions_list, losses_list = [], []
        while os.path.exists(f"offline_data/bvft/{count}.pkl"):
            resolutions, losses = self.toolbox.load_from_pkl(
                f"offline_data/bvft/{count}"
            )
            resolutions_list.append(resolutions)
            losses_list.append(losses)
            count += 1
        for resolutions, losses in zip(resolutions_list, losses_list):
            plt.plot(resolutions, losses)
        plt.legend()
        plt.title("BVFT Losses Where Resolutions Differ")
        plt.xlabel("Resolution")
        plt.ylabel("Loss")
        plt.savefig(f"offline_data/bvft/bvft_losses.pdf")
        plt.close()
        plt.figure(current_figure.number)
