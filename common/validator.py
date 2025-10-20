import multiprocessing
import os
import parser
import time
from multiprocessing import Pool

import numpy as np
import psutil

from ablation_observer import AblationObserver
from global_utils import GeneralUtils
from metric_collector import MetricCollector
from plotter import Plotter
from selection_algorithms import SelectionBaselines


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


def validate_as_single_task(args):
    validator = Validator(misclassified=parser.args.misclassified)
    j_table, policy_indices, create_candidates = args
    return validator.validate(
        j_table=j_table,
        policy_indices=policy_indices,
        create_candidates=create_candidates,
    )


def poll_validate(val_policies_last_time, j_table, num_trials, create_candidates=None):
    if set(val_policies_last_time) == set(parser.completed_policies):
        return val_policies_last_time
    num_cores = available_cpu_cores(percent=10, limit=16)
    multi_args = [(j_table, parser.completed_policies, create_candidates)] * num_trials
    with Pool(processes=num_cores) as pool:
        loggers = list(pool.map(validate_as_single_task, multi_args))

    # Plot results
    plotter = Plotter()
    plotter.plot(loggers)
    return parser.completed_policies


class Validator:
    def __init__(
        self,
        misclassified,
    ):
        self.selector = SelectionBaselines()
        self.toolbox = GeneralUtils()
        self.misclassified = misclassified
        self.max_data_size = parser.args.sampler_size
        self.folder = (
            "misclassified_selections"
            if self.misclassified
            else "realizable_selections"
        )
        # self.toolbox.delete_bvft_cache()

    @staticmethod
    def get_param(env):
        return f"(g={env.unwrapped.model.opt.gravity[2]:.3f}, ε={env.force_scaler:.3f})"

    def selection(
        self, _policy_index, _env_indices, _dataset_index, selection_name, plot_bvft
    ):
        _dataset = self.toolbox.load_dataset(_dataset_index)
        if selection_name == "model_free_bvft":
            return getattr(self.selector, selection_name)(
                policy_index=_policy_index,
                env_indices=_env_indices,
                dataset_index=_dataset_index,
                enable_plot=plot_bvft,
                sample_size=self.max_data_size,
            )
        else:
            return getattr(self.selector, selection_name)(
                policy_index=_policy_index,
                env_indices=_env_indices,
                dataset_index=_dataset_index,
                sample_size=self.max_data_size,
            )

    def validate(
        self, j_table, policy_indices, create_candidates=None, plot_bvft=False
    ):
        """
        env_class should align with the environments created previously, so does policy_class.
        """
        # If realizable cases are considered only, we should take top_k acc into account.
        dataset_indices = parser.args.ground_truth_for_traverse
        env_class, _, _ = self.toolbox.load_env_policy_dataset()
        if create_candidates is None:
            create_candidates = lambda _env_index, _env_class: self.complete(_env_class)
        if not self.misclassified:
            logger = {
                algorithm: {
                    i: {
                        "J Error": float("inf"),
                        f"Top {parser.args.top_k} Accuracy": 0.0,
                    }
                    for i in dataset_indices
                }
                for algorithm in self.selector.baselines
            }
        # If misclassified cases are considered only, then top_k acc just does not make any sense,
        # which should be ignored.
        else:
            logger = {
                algorithm: {i: {"J Error": float("inf")} for i in dataset_indices}
                for algorithm in self.selector.baselines
            }

        # Begin validation
        info = {selection_name: [] for selection_name in self.selector.baselines}
        dataset_indices = parser.args.ground_truth_for_traverse

        # text for plotting
        parser.text_for_comparison = f"Experiment setup:\n"
        for i in dataset_indices:
            candidate_space = create_candidates(i, env_class)
            parser.text_for_comparison += (
                f"target MDP {self.get_param(env_class[i])}, "
                f"candidates {[self.get_param(env_class[t]) for t in candidate_space]}\n"
            )

        for selection_name in self.selector.baselines:
            metric_collector = MetricCollector(
                j_table=j_table,
                policy_indices=policy_indices,
                algorithm_name=selection_name,
                k=-1 if self.misclassified else parser.args.top_k,
            )
            for i in dataset_indices:
                candidate_space = create_candidates(i, env_class)
                for j in policy_indices:
                    # Model selection
                    ranked_env_indices, loss = self.selection(
                        _policy_index=j,
                        _env_indices=candidate_space,
                        _dataset_index=i,
                        selection_name=selection_name,
                        plot_bvft=plot_bvft,
                    )
                    info[selection_name].append(loss)
                    print(ranked_env_indices)
                    os.makedirs(
                        f"offline_data/{self.folder}/{selection_name}/{i}/{j}",
                        exist_ok=True,
                    )
                    self.toolbox.save_as_pkl(
                        f"offline_data/{self.folder}/{selection_name}/{i}/{j}/ranking",
                        ranked_env_indices,
                    )

                    # Performance evaluation for the given selection algorithm
                    metric_collector.update_j_error(
                        env_ground_truth=i,
                        ranked_env_indices=ranked_env_indices,
                        policy_index=j,
                    )
                    if not self.misclassified:
                        metric_collector.update_top_k_indicator(
                            env_ground_truth=i,
                            ranked_env_indices=ranked_env_indices,
                            policy_index=j,
                        )
                (
                    top_k_average_on_policy_class,
                    j_error_average_on_policy_class,
                ) = metric_collector.take_average_on_policy_class(env_index=i)

                logger[selection_name][i]["J Error"] = j_error_average_on_policy_class

                if not self.misclassified:
                    logger[selection_name][i][
                        f"Top {parser.args.top_k} Accuracy"
                    ] = top_k_average_on_policy_class
                    self.toolbox.render_text(
                        f"At MDP {i}, Selection algorithm {selection_name} achieves Top-{parser.args.top_k}-"
                        f"Accuracy {top_k_average_on_policy_class}, J-Error {j_error_average_on_policy_class} "
                        f"on average.",
                        color="YELLOW",
                    )

                else:
                    self.toolbox.render_text(
                        f"At MDP {i}, Selection algorithm {selection_name} achieves J-Error "
                        f"{j_error_average_on_policy_class} on average.",
                        color="YELLOW",
                    )
            parser.text_for_comparison += "\n"
        for selection_name in info:
            loss_list = info[selection_name]
            if selection_name == "trivial_random":
                continue
            # print the average loss with regard to the policy class, on the first MDP.
            loss = [
                sum(loss_list[i][j] for i in range(len(policy_indices)))
                / len(policy_indices)
                for j in range(len(loss_list[0]))
            ]
            print(selection_name, loss)
        return logger

    @staticmethod
    def complete(env_class):
        return list(range(len(env_class)))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser.completed_policies = parser.args.policy_for_traverse
    observer = AblationObserver()
    toolbox = GeneralUtils()
    j_table, l_table = observer.pre_compute(
        num_states=1024, gamma=parser.args.gamma, horizon=observer.INF, enable_pool=True
    )
    val_policies_last_time = poll_validate(
        [], j_table, num_trials=16, create_candidates=None
    )
