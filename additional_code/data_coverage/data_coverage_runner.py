import parser
import time

from env_manager import MDPManager
from global_utils import GeneralUtils, device
from offline_dataset_collector import OfflineDataSetCollector
from rollout_evaluator import RolloutEvaluator

if __name__ == "__main__":
    toolbox = GeneralUtils()
    manager = MDPManager(verbose=True)
    manager.manage()
    # Collect policy class
    (
        index_to_name_and_params,
        mdp_name_to_index,
    ) = toolbox.load_env_class()
    collector = OfflineDataSetCollector(
        index_to_name_and_params=index_to_name_and_params,
        device=device,
        trajectory_max_horizon=parser.args.sampler_max_horizon,
        data_size=parser.args.sampler_size,
        epsilon=parser.args.sampler_epsilon,
        noise_scaler=parser.args.sampler_noise_scaler,
        verbose=parser.args.sampler_verbose,
    )
    assert len(parser.args.policy_for_traverse) == 1
    input(
        f"Target policy: {parser.args.policy_for_traverse}. Press any key for confirmation."
    )
    collector.target_policy_sampler_over_mdp_class(
        target_policy_index=parser.args.policy_for_traverse[0]
    )
    evaluator = RolloutEvaluator(
        gamma=parser.args.gamma,
        max_horizon=parser.args.rollout_max_horizon,
    )
    evaluator.start_rollouts()
    while True:
        bool_completed = toolbox.check_rollouts_completed()
        if not bool_completed:
            print(
                f"Embarrassingly Parallelism has not terminated. Will request for"
                f" validation after {parser.args.request_interval} seconds."
            )
            # 10-seconds interval between each validation request
            time.sleep(parser.args.request_interval)
        else:
            break
