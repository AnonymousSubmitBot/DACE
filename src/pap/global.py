import argparse
import json
import logging
import os
import pickle
import time
from distutils.util import strtobool
from multiprocessing import Manager, Process, Queue
from pathlib import Path

import logzero
import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from ConfigSpace import Configuration
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario

from src.distribution.distributed_master import DistributedMaster
from src.pap import BasePAP
from src.problem_domain import BaseProblem
from src.solver import BaseSolver, BRKGASolver
from src.types_ import *
from src.pap.ceps import evaluate_problem_with_config


class ProblemEvaluator:
    def __init__(self, solver_class: Type[BaseSolver], domain, problem_list, solver_max_eval, max_parallel_num,
                 reference_scores, up_queue, eval_queue, train_num, solver_num, try_index):
        self.solver_class = solver_class
        self.domain = domain
        self.problem_list = problem_list
        self.solver_max_eval = solver_max_eval
        self.max_parallel_num = max_parallel_num
        self.manager = Manager()
        self.reference_scores = reference_scores if reference_scores is not None else [None for _ in self.problem_list]
        self.train_num = train_num
        self.solver_num = solver_num
        self.quality_history = {}
        self.config_known = {}
        self.up_queue = up_queue
        self.eval_queue = eval_queue
        self.try_index = try_index

    def batch_evaluate_problem_with_single_config(self, config: List[Union[float, int, bool]], config_str, per_dict):
        run_num = 0
        result_vector = np.zeros(shape=(len(self.problem_list)))
        for p_index, problem in enumerate(self.problem_list):
            solver = self.solver_class(config=config, max_eval=self.solver_max_eval)
            self.up_queue.put((self.try_index, p_index, problem, solver, self.reference_scores[p_index], False, 1))
            run_num += 1
        for _ in range(run_num):
            result = self.eval_queue.get()
            c_index, p_index, eval_result = result[0], result[1], result[2]
            while c_index != self.try_index:
                self.eval_queue.put(result)
                result = self.eval_queue.get()
                c_index, p_index, eval_result = result[0], result[1], result[2]
            result_vector[p_index] = eval_result
        per_dict[config_str] = result_vector

    def evaluate(self, smac_config: Configuration, seed: int = 1088):
        configs = [smac_config["{}".format(index)] for
                   index in range(len(self.solver_class.config_range) * self.solver_num)]
        config_list = []
        for index in range(self.solver_num):
            config_list.append(
                configs[index * len(self.solver_class.config_range):(index + 1) * len(self.solver_class.config_range)]
            )
        performance_matrix = []
        per_dict = self.manager.dict()
        p_list = []
        eval_config_strs = []
        for c_index, config in enumerate(config_list):
            config_str = "_".join(["{}".format(item) for item in config])
            if config_str not in self.quality_history:
                self.quality_history[config_str] = None
                eval_config_strs.append(config_str)
                p = Process(target=self.batch_evaluate_problem_with_single_config,
                            args=(config, config_str, per_dict))
                p.start()
                p_list.append(p)
        [p.join() for p in p_list]
        for config_str in eval_config_strs:
            result_vector = per_dict[config_str]
            self.quality_history[config_str] = [result_vector[p_index] for p_index in range(len(self.problem_list))]
        for c_index, config in enumerate(config_list):
            config_str = "_".join(["{}".format(item) for item in config])
            new_performance = self.quality_history[config_str]
            performance_matrix.append(np.array(new_performance))
        current_performance = np.max(np.array(performance_matrix), axis=0)
        return np.sum(current_performance)

    def get_performance_matrix(self, config_list):
        performance_matrix = []
        for c_index, config in enumerate(config_list):
            config_str = "_".join(["{}".format(item) for item in config])
            new_performance = self.quality_history[config_str]
            performance_matrix.append(np.array(new_performance))
        return performance_matrix


class GLOBAL(BasePAP):
    def __init__(self, solver_class: Type[BaseSolver], solver_num: int = 4, config_batch_size: int = 100,
                 solver_max_eval: int = 6400, smac_max_try: int = 100, max_parallel_num: int = 60,
                 domain: str = "com_influence_max_problem", train_dim: int = 80, train_num: int = 5,
                 data_root_dir="../../data", distributed: bool = False):
        super().__init__(solver_class, solver_num)
        date_style = '%Y-%m-%d %H:%M:%S'
        file_format = '[%(asctime)s| %(levelname)s |%(filename)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(file_format, date_style)
        self.logger = logzero.setup_logger(logfile=str(Path(os.path.dirname(os.path.abspath(__file__)),
                                                            "../../logs/GLOBAL_logs/global_run_{}_{}.log".format(domain,
                                                                                                                 train_dim))),
                                           formatter=formatter,
                                           name="GLOBAL Log for {} with dim={}".format(domain, train_dim),
                                           level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
                                           maxBytes=int(1e7))
        self.logger.info("START")
        self.config_batch_size: int = config_batch_size
        self.solver_max_eval: int = solver_max_eval
        self.smac_max_try: int = smac_max_try
        self.max_parallel_num = max_parallel_num
        self.domain: str = domain
        self.train_dim: int = train_dim
        self.train_num: int = train_num
        instance_folders = [
            Path(data_root_dir, "problem_instance/train", "{}_{}_{}".format(self.domain, self.train_dim, index)) for
            index in range(self.train_num)]
        self.problem_list = [pickle.load(open(Path(instance_folders[index], "problem.pkl"), "rb")) for index in
                             range(self.train_num)]
        self.reference_scores = [(0, 1) for index in range(self.train_num)]
        self.config_list: List[List[Union[float, int, bool]]] = []
        self.best_performance = -np.inf
        self.manager = Manager()
        self.distributed = distributed
        self.up_queue = Queue()
        self.score_queue = Queue()
        self.eval_queue = Queue()
        if self.distributed:
            self.master = DistributedMaster(up_queue=self.up_queue, score_queue=self.score_queue,
                                            eval_queue=self.eval_queue)
            self.master.start()
            time.sleep(5)
        else:
            self.eval_processes = []
            for p_num in range(self.max_parallel_num):
                p = Process(target=evaluate_problem_with_config,
                            args=(self.up_queue, self.score_queue, self.eval_queue))
                p.start()
                self.eval_processes.append(p)

    def batch_get_problem_reference_score(self, problem_list: List[BaseProblem], p_num=1):
        run_num = 0
        result_vector = [(0, 1) for _ in range(len(problem_list))]
        for p_index, problem in enumerate(problem_list):
            self.up_queue.put(
                (0, p_index, problem, None, None, True, p_num))
            run_num += 1
        for _ in range(run_num):
            result = self.score_queue.get()
            p_index, ref_score = result[0], result[1]
            result_vector[p_index] = ref_score
        return result_vector

    def update_max_parallel_num(self):
        if self.distributed:
            old_num = self.max_parallel_num
            self.max_parallel_num = self.master.get_total_capacity()
            while self.max_parallel_num <= 0:
                self.max_parallel_num = self.master.get_total_capacity()
            if old_num != self.max_parallel_num:
                self.logger.info("The Max Parallel Num is Update to {}".format(self.max_parallel_num))

    def generate_config_space(self):
        cs = ConfigurationSpace(seed=1088)
        index = 0
        for cycle in range(self.solver_num):
            for item in self.solver_class.config_range:
                if item[0] == int:
                    cs.add_hyperparameter(Integer("{}".format(index), (item[1][0], item[1][1]), default=item[2]))
                elif item[0] == float:
                    cs.add_hyperparameter(Float("{}".format(index), (item[1][0], item[1][1]), default=item[2]))
                elif item[0] == bool:
                    cs.add_hyperparameter(Categorical("{}".format(index), [True, False], default=item[2]))
                index += 1
        return cs

    def show_performance(self, performance_matrix):
        for c_index in range(len(self.config_list)):
            self.logger.info("\t".join(["{:.5f}".format(performance_matrix[c_index][p_index]) for p_index in
                                        range(len(self.problem_list))]))
        self.logger.info("Mean Quality\t{:.5f}".format(np.mean(np.max(np.array(performance_matrix), axis=0))))
        self.logger.info("All Quality\t" + "\t".join(
            ["{:.5f}".format(i) for i in np.max(np.array(performance_matrix), axis=0)]))

    def update_config_pop(self):
        self.update_max_parallel_num()
        smac_result = self.manager.dict()
        smac_p_list = []

        def smac_increase(try_index: int, result_dict):
            problem_evaluator = ProblemEvaluator(solver_class=self.solver_class, domain=self.domain,
                                                 problem_list=self.problem_list, solver_max_eval=self.solver_max_eval,
                                                 max_parallel_num=self.max_parallel_num,
                                                 reference_scores=self.reference_scores, train_num=self.train_num,
                                                 solver_num=self.solver_num, up_queue=self.up_queue,
                                                 eval_queue=self.eval_queue, try_index=try_index)
            configspace = self.generate_config_space()
            scenario = Scenario(configspace, n_trials=self.smac_max_try, seed=int(time.time() * 1000) % 10000)
            smac = ACFacade(scenario, problem_evaluator.evaluate, overwrite=True, logging_level=50)
            best_config = smac.optimize()
            configs = [best_config["{}".format(index)] for
                       index in range(len(self.solver_class.config_range) * self.solver_num)]
            config_list = []
            for c_index in range(self.solver_num):
                config_list.append(
                    configs[
                    c_index * len(self.solver_class.config_range):(c_index + 1) * len(self.solver_class.config_range)]
                )
            new_performance_matrix = problem_evaluator.get_performance_matrix(config_list)
            result_dict["SMAC_RESULT_{}".format(try_index)] = (config_list, new_performance_matrix)

        for index in range(self.config_batch_size):
            p = Process(target=smac_increase, args=(index, smac_result))
            p.start()
            smac_p_list.append(p)
        [p.join() for p in smac_p_list]
        update = False
        updated_performance_matrix = []
        per_list = []
        for index in range(self.config_batch_size):
            config_list, new_performance_matrix = smac_result["SMAC_RESULT_{}".format(index)]
            performance = np.sum(np.max(np.array(new_performance_matrix), axis=0))
            per_list.append(float(performance))
            if performance > self.best_performance:
                self.best_performance = performance
                self.config_list = config_list
                update = True
                updated_performance_matrix = new_performance_matrix
        self.logger.info("\t".join(["{:.5f}".format(per / len(self.problem_list)) for per in per_list]))
        if update:
            self.logger.info("The config population has been updated")
            self.show_performance(updated_performance_matrix)
        else:
            self.logger.info("There is no Update.")

    def run(self, time_limit):
        st = time.time()
        self.reference_scores = self.batch_get_problem_reference_score(
            problem_list=self.problem_list, p_num=max(1, self.max_parallel_num // len(self.problem_list)))

        experiment_record = []
        while time.time() - st < time_limit * 3600:
            self.update_config_pop()
            experiment_record.append(self.config_list)
            pickle.dump(experiment_record,
                        open("../../logs/GLOBAL_logs/{}_{}.pkl".format(self.domain, self.train_dim), "wb"))
        self.logger.info("Finish Construction")

    def terminate(self):
        if self.distributed:
            self.master.close_evaluator()
            self.master.terminate_process()
            self.master.kill()
        else:
            [p.kill() for p in self.eval_processes if p.is_alive()]


if __name__ == '__main__':
    # You should create a tmp dir which mount a tmpfs (MEM DISK) and link it to `src/pap/smac3_output`
    # to accelerate the DISK IO, which is caused by the SMAC3 package must use file to storage
    # intermediate file. If you don't use MEM DISK, the io wait will increase quickly and the
    # use ratio of CPUs will decrease.

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_domain', type=str, default="contamination_problem",
                        help='Optimization Problem Domain')
    parser.add_argument('--solver_num', type=int, default=4, help='Number of Solvers in PAP')
    parser.add_argument('--problem_dim', type=int, default=30, help='Optimization Problem Dimension')
    parser.add_argument('--solver_max_eval', type=int, default=800, help='Max Eval Times for each Member Optimizer')
    parser.add_argument('--config_batch_size', type=int, default=20, help='Batch Size while Search Config Pop')
    parser.add_argument('--smac_max_try', type=int, default=6400, help='Max Eval Times for SMAC')
    parser.add_argument('--max_parallel_num', type=int, default=300,
                        help='Max Parallel Processes for Evaluator on this machine')
    parser.add_argument('--distributed', type=strtobool, default=False, help='Run GLOBAL in Distributed Mode')
    parser.add_argument('--time_limit', type=float, default=7, help='Time limit, unit is hour')
    args = parser.parse_args()

    glo = GLOBAL(solver_class=BRKGASolver, config_batch_size=args.config_batch_size, solver_num=args.solver_num,
                 solver_max_eval=args.solver_max_eval, smac_max_try=args.smac_max_try,
                 domain=args.problem_domain, train_dim=args.problem_dim,
                 max_parallel_num=args.max_parallel_num, distributed=args.distributed, )
    glo.logger.info("GLOBAL Configuration is: {}".format(json.dumps(vars(args), ensure_ascii=False, indent=2)))
    glo.run(args.time_limit)
    glo.terminate()
