import multiprocessing
import parser
import time

from ablation_observer import AblationObserver
from env_manager import MDPManager
from global_utils import GeneralUtils, device
from offline_dataset_collector import OfflineDataSetCollector
from policy_trainer import PolicyTrainer
from rollout_evaluator import RolloutEvaluator
from validator import Validator, poll_validate


def create_candidates(env_index, env_class, gap=10.0):
    return list(range(len(env_class)))


class Runner:
    def __init__(self):
        self.toolbox = GeneralUtils()
        self.toolbox.delete_bvft_cache()

    @staticmethod
    def run():
        # Toolbox preparation
        toolbox = GeneralUtils()
        toolbox.fix_all()
        if not parser.args.jump_to_validation:
            # Collect MDP class
            manager = MDPManager(verbose=True)
            manager.manage()

            # Collect policy class
            (
                index_to_name_and_params,
                mdp_name_to_index,
            ) = toolbox.load_env_class()
            trainer = PolicyTrainer(
                index_to_name_and_params=index_to_name_and_params,
                epsilon=parser.args.trainer_epsilon,
                total_steps=parser.args.trainer_total_steps,
                checkpoint=parser.args.trainer_checkpoint,
                lr=parser.args.lr,
                hidden_layers=parser.args.hidden_layers,
                limit=parser.args.trainer_limit,
                algorithm=parser.args.trainer_algorithm,
                device=device,
                verbose=True,
                clear_ckpt_after_train=parser.args.clear_ckpt_after_train,
            )
            trainer.train_from_checkpoints()

            # Collect offline dataset
            collector = OfflineDataSetCollector(
                index_to_name_and_params=index_to_name_and_params,
                device=device,
                trajectory_max_horizon=parser.args.sampler_max_horizon,
                data_size=parser.args.sampler_size,
                epsilon=parser.args.sampler_epsilon,
                noise_scaler=parser.args.sampler_noise_scaler,
                verbose=parser.args.sampler_verbose,
            )
            collector.epsilon_greedy_sampler_over_mdp_class()

            # Ablation study - stage 1
            # observer.early_observe()

            observer = AblationObserver()
            j_table, l_table = observer.pre_compute(
                num_states=1024,
                gamma=parser.args.gamma,
                horizon=observer.INF,
                enable_pool=True,
            )
            observer.early_observe()
            evaluator = RolloutEvaluator(
                gamma=parser.args.gamma,
                max_horizon=parser.args.rollout_max_horizon,
            )

            evaluator.start_rollouts()
        val_policies_last_time = []

        observer = AblationObserver()
        j_table, l_table = observer.pre_compute(
            num_states=1024,
            gamma=parser.args.gamma,
            horizon=observer.INF,
            enable_pool=True,
        )
        observer.early_observe()
        while True:
            bool_completed = toolbox.check_rollouts_completed()

            val_policies_last_time = poll_validate(
                val_policies_last_time,
                j_table,
                num_trials=16,
                create_candidates=create_candidates,
            )
            print(f"Current completed policies: {val_policies_last_time}")
            if not bool_completed:
                print(
                    f"Embarrassingly Parallelism has not terminated. Will request for"
                    f" validation after {parser.args.request_interval} seconds."
                )
                # 10-seconds interval between each validation request
                time.sleep(parser.args.request_interval)
            else:
                break

        observer.afterward_observe()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    runner = Runner()
    runner.run()
