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
from ConfigSpace import Configuration
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario

from src.distribution.distributed_master import DistributedMaster
from src.mutation import CEPSMutator
from src.pap import BasePAP
from src.problem_domain import BaseProblem
from src.solver import BaseSolver, BRKGASolver
from src.types_ import *


def evaluate_problem_with_config(up_queue=None, score_queue=None, eval_queue=None, distributed=False):
    while True:
        data = up_queue.get()
        c_index, p_index, problem, solver, ref_score, get_score = data[0], data[1], data[2], data[3], data[4], data[5]
        if get_score:
            ref_solver = BRKGASolver(config=[20, 70, 10, 0.7, False], max_eval=800)
            max_y = ref_solver.optimize(problem)[1]
            min_y = ref_solver.optimize(problem, low_better=True)[1]
            ref_score = (min_y, max_y)
            score_queue.put(
                (p_index, ref_score) if not distributed else (p_index, ref_score, data[-1]))
        else:
            if ref_score is None:
                ref_solver = BRKGASolver(config=[20, 70, 10, 0.7, False], max_eval=800)
                max_y = ref_solver.optimize(problem)[1]
                min_y = ref_solver.optimize(problem, low_better=True)[1]
                ref_score = (min_y, max_y)
            eval_result = solver.optimize(problem)[1]
            eval_result = (eval_result - ref_score[0]) / (ref_score[1] - ref_score[0])
            eval_queue.put(
                (c_index, p_index, eval_result) if not distributed else (c_index, p_index, eval_result, data[-1]))


class ProblemEvaluator:
    def __init__(self, solver_class: Type[BaseSolver], domain, problem_list, solver_max_eval, max_parallel_num,
                 process_list, reference_scores, current_config_list, performance_matrix, up_queue, eval_queue,
                 try_index):
        self.solver_class = solver_class
        self.domain = domain
        self.problem_list = problem_list
        self.solver_max_eval = solver_max_eval
        self.max_parallel_num = max_parallel_num
        self.manager = Manager()
        self.process_list = process_list
        self.reference_scores = reference_scores if reference_scores is not None else [None for _ in self.problem_list]
        self.current_config_list = current_config_list
        self.performance_matrix = performance_matrix
        self.quality_history = {}
        self.config_known = {}
        self.up_queue = up_queue
        self.eval_queue = eval_queue
        self.try_index = try_index

    def batch_evaluate_problem_with_single_config(self, config: List[Union[float, int, bool]]):
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
        return result_vector

    def evaluate(self, smac_config: Configuration, seed: int = 1088):
        config = [smac_config["{}".format(index)] for index in range(len(self.solver_class.config_range))]
        config_str = "_".join(["{}".format(item) for item in config])
        if config_str not in self.quality_history:
            result_vector = self.batch_evaluate_problem_with_single_config(config=config)
            self.quality_history[config_str] = [result_vector[p_index] for p_index in range(len(self.problem_list))]
            self.config_known[config_str] = config  # print(config_str, self.quality_history[config_str])
        new_performance = self.quality_history[config_str]
        current_performance = np.max(np.array(self.performance_matrix), axis=0)
        improve = np.sum(
            [max(0, new_performance[index] - current_performance[index]) for index in range(len(self.problem_list))])
        if improve == 0:
            improve = np.min(
                [new_performance[index] - current_performance[index] for index in range(len(self.problem_list))])
        return improve

    def get_best_config_list(self):
        current_performance = np.max(np.array(self.performance_matrix), axis=0)
        candidate_config_str = list(self.quality_history.keys())
        best_config_str = max(candidate_config_str, key=lambda i: np.sum(
            [max(current_performance[p_index], self.quality_history[i][p_index]) for p_index in
             range(len(self.problem_list))]))
        new_config = self.config_known[best_config_str]
        new_performance = self.quality_history[best_config_str]

        new_config_list = self.current_config_list.copy()
        new_performance_matrix = self.performance_matrix.copy()
        new_performance_matrix.append(new_performance)
        problem_performance = np.max(np.array(new_performance_matrix), axis=0)
        new_config_list.append(new_config)
        return np.sum(problem_performance), new_config_list, new_performance_matrix, new_config, new_performance


class CEPS(BasePAP):
    def __init__(self, solver_class: Type[BaseSolver], solver_num: int = 4, max_iter: int = 4, temp_num: int = 4,
                 solver_max_eval: int = 6400, smac_max_try: int = 100, mutator_epoch_num: int = 100,
                 mutator_pop_size: int = 99, max_parallel_num: int = 60, domain: str = "com_influence_max_problem",
                 fixed_initial_config_set: bool = True, train_dim: int = 80, train_num: int = 5,
                 data_root_dir="../../data", distributed: bool = False, add_recommend_config_set=False):
        super().__init__(solver_class, solver_num)
        date_style = '%Y-%m-%d %H:%M:%S'
        file_format = '[%(asctime)s| %(levelname)s |%(filename)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(file_format, date_style)
        self.logger = logzero.setup_logger(logfile=str(Path(os.path.dirname(os.path.abspath(__file__)),
                                                            "../../logs/CEPS_logs/ceps_run_{}_{}.log".format(domain,
                                                                                                             train_dim))),
                                           formatter=formatter,
                                           name="CEPS Log for {} with dim={}".format(domain, train_dim),
                                           level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
                                           maxBytes=int(1e7))
        self.logger.info("START")
        self.max_iter: int = max_iter
        self.temp_num: int = temp_num
        self.solver_max_eval: int = solver_max_eval
        self.smac_max_try: int = smac_max_try
        self.mutator_epoch_num = mutator_epoch_num
        self.mutator_pop_size = mutator_pop_size
        self.max_parallel_num = max_parallel_num
        self.domain: str = domain
        self.train_dim: int = train_dim
        self.train_num: int = train_num
        self.fixed_initial_config_set = fixed_initial_config_set
        self.add_recommend_config_set = add_recommend_config_set
        instance_folders = [
            Path(data_root_dir, "problem_instance/train", "{}_{}_{}".format(self.domain, self.train_dim, index)) for
            index in range(self.train_num)]
        self.problem_list = [pickle.load(open(Path(instance_folders[index], "problem.pkl"), "rb")) for index in
                             range(self.train_num)]
        scores = [np.load(open(Path(instance_folders[index], "y.npy"), "rb")) for index in range(self.train_num)]
        self.reference_scores = [(np.min(scores[index]), np.max(scores[index])) for index in range(self.train_num)]
        self.config_list: List[List[Union[float, int, bool]]] = []
        self.performance_matrix = [[-np.inf for _ in range(self.train_num)] for _ in range(self.solver_num)]
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

    def batch_evaluate_problem_with_config_list(self, config_list: List[List[Union[float, int, bool]]],
                                                problem_list: List[BaseProblem], reference_scores=None) -> Union[
        NpArray, List[List[float]]]:
        run_num = 0
        result_matrix = np.zeros(shape=(len(config_list), len(problem_list)))
        reference_scores = [None for _ in problem_list] if reference_scores is None else reference_scores
        for p_index, problem in enumerate(problem_list):
            for c_index, config in enumerate(config_list):
                solver = self.solver_class(config=config, max_eval=self.solver_max_eval)
                self.up_queue.put((c_index, p_index, problem, solver, reference_scores[p_index], False,
                                   max(1, self.max_parallel_num // len(problem_list))))
                run_num += 1
        for _ in range(run_num):
            result = self.eval_queue.get()
            c_index, p_index, eval_result = result[0], result[1], result[2]
            result_matrix[c_index][p_index] = eval_result
        return result_matrix

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

    def get_problem_quality(self, problem: BaseProblem) -> float:
        result_matrix = self.batch_evaluate_problem_with_config_list(config_list=self.config_list,
                                                                     problem_list=[problem])
        return np.max([result_matrix[c_index][0] for c_index in range(len(self.config_list))])

    def get_problem_batch_quality(self, problem_list: List[BaseProblem], reference_scores=None) -> NpArray:
        result_matrix = self.batch_evaluate_problem_with_config_list(config_list=self.config_list,
                                                                     problem_list=problem_list,
                                                                     reference_scores=reference_scores)
        candidate_performance_matrix = [[result_matrix[c_index][p_index] for p_index in range(len(problem_list))] for
                                        c_index in range(len(self.config_list))]
        candidate_performance = np.max(np.array(candidate_performance_matrix), axis=0)
        return candidate_performance

    def get_config_list_performance(self, config_list: List[List[Union[float, int, bool]]]) -> List[float]:
        result_matrix = self.batch_evaluate_problem_with_config_list(config_list=config_list,
                                                                     problem_list=self.problem_list,
                                                                     reference_scores=self.reference_scores)
        return [max([result_matrix[c_index][p_index] for c_index in range(len(self.config_list))]) for p_index in
                range(len(self.problem_list))]

    def update_performance_matrix(self):
        self.update_max_parallel_num()
        self.reference_scores = self.batch_get_problem_reference_score(
            problem_list=self.problem_list, p_num=max(1, self.max_parallel_num // len(self.problem_list)))
        result_matrix = self.batch_evaluate_problem_with_config_list(config_list=self.config_list,
                                                                     problem_list=self.problem_list,
                                                                     reference_scores=self.reference_scores)
        self.performance_matrix = [[result_matrix[c_index][p_index] for p_index in range(len(self.problem_list))] for
                                   c_index in range(len(self.config_list))]

    def update_max_parallel_num(self):
        if self.distributed:
            old_num = self.max_parallel_num
            self.max_parallel_num = self.master.get_total_capacity()
            while self.max_parallel_num <= 0:
                self.max_parallel_num = self.master.get_total_capacity()
            if old_num != self.max_parallel_num:
                self.logger.info("The Max Parallel Num is Update to {}".format(self.max_parallel_num))

    def show_performance(self):
        for c_index in range(len(self.config_list)):
            self.logger.info("\t".join(["{:.5f}".format(self.performance_matrix[c_index][p_index]) for p_index in
                                        range(len(self.problem_list))]))
        self.logger.info("Mean Quality\t{:.5f}".format(np.mean(np.max(np.array(self.performance_matrix), axis=0))))
        self.logger.info("All Quality\t" + "\t".join(
            ["{:.5f}".format(i) for i in np.max(np.array(self.performance_matrix), axis=0)]))

    def initialization(self, config_sample_num: int = 50) -> Tuple[List[List[Union[float, int, bool]]], List[float]]:
        self.update_max_parallel_num()
        self.config_list = []
        if self.fixed_initial_config_set:
            c: List[List[Union[float, int, bool]]] = pickle.load(
                open(Path(os.path.dirname(os.path.abspath(__file__)),
                          "../../logs/initial_config_sets/brgka_configs_{}.pkl".format(config_sample_num)),
                     "rb"))
        else:
            c: List[List[Union[float, int, bool]]] = [self.sample_config() for _ in range(config_sample_num)]
        if self.add_recommend_config_set:
            c: List[List[Union[float, int, bool]]] = c[:-4] + self.solver_class.recommend_config
        self.reference_scores = self.batch_get_problem_reference_score(
            problem_list=self.problem_list, p_num=max(1, self.max_parallel_num // len(self.problem_list)))
        result_matrix = self.batch_evaluate_problem_with_config_list(config_list=c, problem_list=self.problem_list,
                                                                     reference_scores=self.reference_scores)
        ins_performances: List[float] = [-np.inf for _ in self.problem_list]
        candidate_c_indices: List[int] = [i for i in range(len(c))]
        for _ in range(self.solver_num):
            select_c = max(candidate_c_indices, key=lambda i: np.sum(
                [max(ins_performances[p_index], result_matrix[i][p_index]) for p_index in
                 range(len(self.problem_list))]))
            candidate_c_indices.remove(select_c)
            self.config_list.append(c[select_c])
            ins_performances = [max(ins_performances[p_index], result_matrix[select_c][p_index]) for p_index in
                                range(len(self.problem_list))]
            for p_index in range(len(self.problem_list)):
                self.performance_matrix[len(self.config_list) - 1][p_index] = result_matrix[select_c][p_index]
        return self.config_list, ins_performances

    def update_config_pop(self, random_del=False):
        self.update_max_parallel_num()
        process_list = self.manager.list()
        smac_result = self.manager.dict()
        smac_p_list = []

        def smac_increase(try_index: int, result_dict):
            del_index = np.random.randint(0, len(self.config_list) - 1) if random_del else try_index % len(
                self.config_list)
            new_config_list = self.config_list[:del_index] + self.config_list[del_index + 1:]
            new_performance_matrix = self.performance_matrix[:del_index] + self.performance_matrix[del_index + 1:]
            problem_evaluator = ProblemEvaluator(solver_class=self.solver_class, domain=self.domain,
                                                 problem_list=self.problem_list, solver_max_eval=self.solver_max_eval,
                                                 max_parallel_num=self.max_parallel_num, process_list=process_list,
                                                 reference_scores=self.reference_scores,
                                                 current_config_list=new_config_list,
                                                 performance_matrix=new_performance_matrix, up_queue=self.up_queue,
                                                 eval_queue=self.eval_queue, try_index=try_index)
            configspace = self.generate_config_space()
            scenario = Scenario(configspace, n_trials=self.smac_max_try, seed=int(time.time() * 1000) % 10000)
            smac = ACFacade(scenario, problem_evaluator.evaluate, overwrite=True, logging_level=50)
            best_config = smac.optimize()
            new_list_quality, new_list, new_matrix, new_config, new_performance = problem_evaluator.get_best_config_list()
            result_dict["SMAC_RESULT_{}".format(try_index)] = (
                new_list_quality, new_list, new_matrix, new_config, new_performance)

        for index in range(self.temp_num):
            p = Process(target=smac_increase, args=(index, smac_result))
            p.start()
            smac_p_list.append(p)
        [p.join() for p in smac_p_list]
        candidate_configs = self.config_list.copy()
        candidate_matrix = self.performance_matrix.copy()
        for index in range(self.temp_num):
            candidate_configs.append(smac_result["SMAC_RESULT_{}".format(index)][3])
            candidate_matrix.append(smac_result["SMAC_RESULT_{}".format(index)][4])
        _, max_indices = self.exhaustive_search_pap(candidate_configs, candidate_matrix)
        self.config_list = [candidate_configs[index] for index in max_indices]
        self.performance_matrix = [candidate_matrix[c_index] for c_index in max_indices]

    def update_problem_pop(self):
        self.update_max_parallel_num()
        reserved_indices = list(range(len(self.problem_list)))
        added_problems = []
        current_performance = np.max(np.array(self.performance_matrix), axis=0)
        break_check_value = (np.min(current_performance), np.mean(current_performance), np.max(current_performance))
        break_check_step = (self.mutator_epoch_num // 4, self.mutator_epoch_num // 2, self.mutator_epoch_num)
        while True:
            mutator = CEPSMutator(self.train_dim, self.domain, popsize=self.mutator_pop_size)
            hardest_problem, hardest_quality = None, np.inf
            start_index = 0
            if self.domain == "compiler_args_selection_problem":
                start_index = break_check_step[0]
                self.update_max_parallel_num()
                self.logger.info(
                    "For the Compiler Args Selection Problem, Evaluate First {} Batch mutation ".format(
                        start_index) + "result parallel to accelerate")
                candidate_problems_batch = [mutator.ask() for _ in range(start_index)]
                candidate_problems = [item for sublist in candidate_problems_batch for item in sublist]
                candidate_reference_scores = self.batch_get_problem_reference_score(problem_list=candidate_problems,
                                                                                    p_num=1)
                candidate_performance = self.get_problem_batch_quality(problem_list=candidate_problems,
                                                                       reference_scores=candidate_reference_scores)
                best_index = np.argmin(candidate_performance)
                if candidate_performance[best_index] < hardest_quality:
                    hardest_problem, hardest_quality = candidate_problems[best_index], candidate_performance[best_index]
                self.logger.info("{:.5f} {:.5f} {:.5f}".format(hardest_quality, np.min(candidate_performance),
                                                               np.mean(candidate_performance)))
                self.logger.info("Continue with the {}th epoch".format(start_index + 1))

            for es_epoch in range(start_index, self.mutator_epoch_num):
                if np.sum([1 if es_epoch >= break_check_step[i] and hardest_quality <= break_check_value[i] else 0 for i
                           in range(3)]) > 0:
                    break
                self.update_max_parallel_num()
                candidate_problems = mutator.ask()
                candidate_reference_scores = self.batch_get_problem_reference_score(
                    problem_list=candidate_problems, p_num=max(1, self.max_parallel_num // len(candidate_problems)))
                candidate_performance = self.get_problem_batch_quality(problem_list=candidate_problems,
                                                                       reference_scores=candidate_reference_scores)
                best_index = np.argmin(candidate_performance)
                if candidate_performance[best_index] < hardest_quality:
                    hardest_problem, hardest_quality = candidate_problems[best_index], candidate_performance[best_index]
                mutator.tell(-candidate_performance)
                self.logger.info("{:.5f} {:.5f} {:.5f}".format(hardest_quality, np.min(candidate_performance),
                                                               np.mean(candidate_performance)))
            easier_indices = [i for i in reserved_indices if current_performance[i] > hardest_quality]
            if len(easier_indices) == 0:
                break
            else:
                remove_index = np.random.choice(easier_indices)
                reserved_indices.remove(remove_index)
                added_problems.append(hardest_problem)
            self.logger.info("{} {} {}".format(reserved_indices, len(added_problems), len(easier_indices)))
            if len(reserved_indices) == 0:
                break
            if len(added_problems) > len(self.problem_list) // 2 or len(added_problems) > 2 * self.train_num:
                break
        self.problem_list = added_problems + self.problem_list
        self.update_performance_matrix()

    def run(self, load_pickle: bool = False):
        self.update_max_parallel_num()
        st = time.time()
        if load_pickle:
            experiment_record = pickle.load(
                open("../../logs/CEPS_logs/{}_{}.pkl".format(self.domain, self.train_dim), "rb"))
            self.config_list = experiment_record["initial_configs"]
            self.problem_list = experiment_record["initial_problems"]

            if len(experiment_record["update_history"]) > 0:
                self.config_list = experiment_record["update_history"][-1]["config_pop"]
                if "problem_pop" in experiment_record["update_history"][-1].keys():
                    self.problem_list = experiment_record["update_history"][-1]["problem_pop"]
                elif len(experiment_record["update_history"]) > 1:
                    self.problem_list = experiment_record["update_history"][-2]["problem_pop"]
            self.update_performance_matrix()
            self.show_performance()
        else:
            experiment_record = {}
            self.initialization(config_sample_num=50)
            self.logger.info(time.time() - st)
            self.show_performance()

            experiment_record["initial_configs"] = self.config_list
            experiment_record["initial_problems"] = self.problem_list
            experiment_record["update_history"] = []
            pickle.dump(experiment_record,
                        open("../../logs/CEPS_logs/{}_{}.pkl".format(self.domain, self.train_dim), "wb"))

        current_iter = len(experiment_record["update_history"])

        if current_iter > 0 and "problem_pop" not in experiment_record["update_history"][-1].keys():
            current_iter -= 1

        for iter_num in range(current_iter, self.max_iter):
            if len(experiment_record["update_history"]) == iter_num:
                self.logger.info("START Config Update for iter {}".format(iter_num + 1))
                self.update_config_pop()
                self.logger.info("{} {} {}".format(iter_num + 1, time.time() - st, "Finish Config Population Update"))
                self.show_performance()

                experiment_record["update_history"].append({"config_pop": self.config_list, })

            pickle.dump(experiment_record,
                        open("../../logs/CEPS_logs/{}_{}.pkl".format(self.domain, self.train_dim), "wb"))

            if iter_num != self.max_iter - 1:
                self.logger.info("START Problem Update for iter {}".format(iter_num + 1))
                self.update_problem_pop()
                self.logger.info("{} {} {}".format(iter_num + 1, time.time() - st, "Finish Problem Population Update"))
                self.show_performance()
                experiment_record["update_history"][iter_num]["problem_pop"] = self.problem_list
            pickle.dump(experiment_record,
                        open("../../logs/CEPS_logs/{}_{}.pkl".format(self.domain, self.train_dim), "wb"))

    def only_config_update(self, load_pickle: bool = False):
        st = time.time()
        if load_pickle:
            experiment_record = pickle.load(
                open("../../logs/CEPS_logs/{}_{}.pkl".format(self.domain, self.train_dim), "rb"))
            self.config_list = experiment_record["initial_configs"]
            self.problem_list = experiment_record["initial_problems"]
            self.update_performance_matrix()
        else:
            experiment_record = {}
            self.initialization(config_sample_num=50)
            self.show_performance()
            experiment_record["initial_configs"] = self.config_list
            experiment_record["initial_problems"] = self.problem_list
        experiment_record["update_history"] = []
        pickle.dump(experiment_record,
                    open("../../logs/CEPS_logs/{}_{}_only_update_config.pkl".format(self.domain, self.train_dim), "wb"))
        for iter_num in range(self.max_iter):
            self.logger.info("START Config Update for iter {}".format(iter_num + 1))
            self.update_config_pop()
            self.logger.info("{} {} {}".format(iter_num + 1, time.time() - st, "Finish Config Population Update"))
            self.show_performance()

            experiment_record["update_history"].append({"config_pop": self.config_list, })
            pickle.dump(experiment_record,
                        open("../../logs/DACE_logs/{}_{}_only_update_config.pkl".format(self.domain, self.train_dim),
                             "wb"))

    def terminate(self):
        if self.distributed:
            self.master.close_evaluator()
            self.master.terminate_process()
            self.master.kill()
        else:
            [p.kill() for p in self.eval_processes if p.is_alive()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_domain', type=str, default="contamination_problem",
                        help='Optimization Problem Domain')
    parser.add_argument('--problem_dim', type=int, default=30, help='Optimization Problem Dimension')
    parser.add_argument('--max_iter', type=int, default=4, help='Max Iteration Times for CEPS')
    parser.add_argument('--solver_max_eval', type=int, default=800, help='Max Eval Times for each Member Optimizer')
    parser.add_argument('--solver_num', type=int, default=4, help='Number of Solvers in PAP')
    parser.add_argument('--temp_num', type=int, default=20, help='Temp PAP while Update Config Pop')
    parser.add_argument('--smac_max_try', type=int, default=1600, help='Max Eval Times for SMAC')
    parser.add_argument('--mutator_epoch_num', type=int, default=200, help='Max Epoch for Problem Mutation')
    parser.add_argument('--mutator_pop_size', type=int, default=81, help='Pop for each Epoch for Problem Mutation')
    parser.add_argument('--max_parallel_num', type=int, default=300,
                        help='Max Parallel Processes for Evaluator on this machine')
    parser.add_argument('--fixed_initial_config_set', type=strtobool, default=True,
                        help='Use a Fixed Initial Config Set C')
    parser.add_argument('--add_recommend_config_set', type=strtobool, default=False,
                        help='Add Recommend Default Configs into Initial Config Set C')
    parser.add_argument('--distributed', type=strtobool, default=False, help='Run CEPS in Distributed Mode')
    parser.add_argument('--load_pickle', type=strtobool, default=False, help='Load Pickle From Last Run')
    parser.add_argument('--only_config_update', type=strtobool, default=False,
                        help='Generate new Problem Instance or not')
    args = parser.parse_args()

    ceps = CEPS(solver_class=BRKGASolver, solver_max_eval=args.solver_max_eval, smac_max_try=args.smac_max_try,
                temp_num=args.temp_num, domain=args.problem_domain, train_dim=args.problem_dim,
                max_parallel_num=args.max_parallel_num, mutator_epoch_num=args.mutator_epoch_num,
                mutator_pop_size=args.mutator_pop_size, max_iter=args.max_iter, distributed=args.distributed,
                add_recommend_config_set=args.add_recommend_config_set, solver_num=args.solver_num,
                fixed_initial_config_set=args.fixed_initial_config_set)
    ceps.logger.info("CEPS Configuration is: {}".format(json.dumps(vars(args), ensure_ascii=False, indent=2)))
    if args.only_config_update:
        ceps.only_config_update(load_pickle=args.load_pickle)
    else:
        ceps.run(load_pickle=args.load_pickle)
    ceps.terminate()
