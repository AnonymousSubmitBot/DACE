import copy
import os
import pickle
import time
from itertools import combinations
from multiprocessing import Manager, Process, Queue
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from src.problem_domain import BaseProblem
from src.solver.base_solver import BaseSolver
from src.distribution import DistributedMaster
from src.types_ import *


def single_eval(up_queue: Queue, score_queue: Queue, eval_queue: Queue, distributed=False):
    while True:
        data = up_queue.get()
        c_index, p_index, try_index, solver, problem, seed = data[0], data[1], data[2], data[3], data[4], data[5]
        result = solver.optimize(
            problem_instance=copy.deepcopy(problem),
            seed=seed
        )
        if not distributed:
            eval_queue.put((c_index, p_index, try_index, result[1], result[2]))
        else:
            eval_queue.put((c_index, p_index, try_index, result[1], result[2], data[-1]))


class BasePAP(object):
    def __init__(self, solver_class: Type[BaseSolver], solver_num: int = 4, max_eval: int = 3200):
        self.solver_class = solver_class
        self.solver_num = solver_num
        self.max_eval = max_eval
        self.config_list: List[List[Union[float, int, bool]]] = []

    def exhaustive_search_pap(self, candidate_configs, candidate_matrix):
        candidate_indices = list(range(len(candidate_configs)))
        max_performance, max_indices = -np.inf, list(range(self.solver_num))
        for pap_indices in combinations(candidate_indices, self.solver_num):
            performance = np.sum([np.max([candidate_matrix[c_index][p_index] for c_index in pap_indices]) for p_index in
                                  range(len(candidate_matrix[0]))])
            if performance > max_performance:
                max_performance, max_indices = performance, pap_indices
        return max_performance, max_indices

    def sample_config(self) -> List[Union[float, int, bool]]:
        config: List[Union[float, int, bool]] = []
        for item in self.solver_class.config_range:
            if item[0] == int:
                config.append(np.random.randint(item[1][0], item[1][1]))
            elif item[0] == float:
                config.append(np.random.random() * item[1][1] - item[1][0])
            elif item[0] == bool:
                config.append(True if np.random.random() > 0.5 else False)
        return config

    def generate_config_space(self):
        cs = ConfigurationSpace(seed=1088)
        for index, item in enumerate(self.solver_class.config_range):
            if item[0] == int:
                cs.add_hyperparameter(Integer("{}".format(index), (item[1][0], item[1][1]), default=item[2]))
            elif item[0] == float:
                cs.add_hyperparameter(Float("{}".format(index), (item[1][0], item[1][1]), default=item[2]))
            elif item[0] == bool:
                cs.add_hyperparameter(Categorical("{}".format(index), [True, False], default=item[2]))
        return cs

    def set_config_list(self, config_list):
        self.config_list = config_list

    def evaluate_parallel(self, problem_list: List[BaseProblem], max_parallel_num=60, seed=1088, device_index=None,
                          up_queue: Queue = None, eval_queue: Queue = None, try_index: int = 0, result_dict=None):
        eval_processes = []
        if up_queue is None or eval_queue is None:
            up_queue = Queue()
            eval_queue = Queue()
            for _ in range(max_parallel_num):
                p = Process(target=single_eval,
                            args=(up_queue, None, eval_queue))
                p.start()
                eval_processes.append(p)
        run_num = 0
        for c_index, config in enumerate(self.config_list):
            for p_index, problem in enumerate(problem_list):
                if device_index is not None:
                    solver = self.solver_class(config, max_eval=self.max_eval, device_index=device_index)
                else:
                    solver = self.solver_class(config, max_eval=self.max_eval)
                up_queue.put((c_index, p_index, try_index, solver, problem, seed))
                run_num += 1
        performance_matrix = np.zeros(shape=(len(self.config_list), len(problem_list)))
        step_history_matrix = np.zeros(shape=(len(self.config_list), len(problem_list), self.max_eval))
        for _ in range(run_num):
            result = eval_queue.get()
            c_index, p_index, return_index = result[0], result[1], result[2]
            while return_index != try_index:
                eval_queue.put(result)
                result = eval_queue.get()
                c_index, p_index, return_index = result[0], result[1], result[2]
            performance_matrix[c_index][p_index] = result[3]
            step_history_matrix[c_index][p_index] = result[4][:self.max_eval]
        [p.kill() for p in eval_processes if p.is_alive()]
        performances = np.max(performance_matrix, axis=0)
        all_performance = np.sum(performances)
        if result_dict is not None:
            result_dict[try_index] = (all_performance, performances, performance_matrix, step_history_matrix)
        return all_performance, performances, performance_matrix, step_history_matrix

    def eval_parallel_multi_times(self, problem_list: List[BaseProblem], max_parallel_num=60, eval_times: int = 5,
                                  load_pickle_path: str = None, device_index=None, up_queue: Queue = None,
                                  eval_queue: Queue = None, distributed=False):
        seed_set = [1088, 1024, 2048, 2333, 3306, 3389, 443, 9300, 6379, 8080, 9200, 5432]
        all_result = []
        eval_processes = []
        master = None
        if up_queue is None or eval_queue is None:
            up_queue = Queue()
            eval_queue = Queue()
            if distributed:
                master = DistributedMaster(up_queue=up_queue,
                                           eval_queue=eval_queue)
                master.start()
                time.sleep(5)
            else:
                for _ in range(max_parallel_num):
                    p = Process(target=single_eval,
                                args=(up_queue, None, eval_queue))
                    p.start()
                    eval_processes.append(p)
        if load_pickle_path is not None and os.path.exists(load_pickle_path):
            all_result = pickle.load(open(load_pickle_path, "rb"))
        all_performance = sum([result["all_perf"] for result in all_result])
        performances = sum([result["perfs"] for result in all_result])
        performance_matrix = sum([result["perf_matrix"] for result in all_result])
        eval_parallel_processes = []
        manager = Manager()
        result_dict = manager.dict()
        for index in range(len(all_result), eval_times):
            seed = seed_set[index % len(seed_set)] * ((eval_times // len(seed_set)) + 1)
            p = Process(target=self.evaluate_parallel,
                        args=(
                            problem_list, max_parallel_num, seed, device_index, up_queue, eval_queue, index,
                            result_dict))
            p.start()
            eval_parallel_processes.append(p)
        [p.join() for p in eval_parallel_processes]
        for index in range(len(all_result), eval_times):
            all_perf, perfs, perf_matrix, step_history_matrix = result_dict[index]
            all_performance += all_perf
            performances += perfs
            performance_matrix += perf_matrix
            all_result.append({
                "seed": seed_set[index % len(seed_set)] * ((eval_times // len(seed_set)) + 1),
                "step_history_matrix": step_history_matrix,
                "all_perf": all_perf,
                "perfs": perfs,
                "perf_matrix": perf_matrix
            })
            if load_pickle_path is not None:
                pickle.dump(all_result, open(load_pickle_path, "wb"))
        if distributed and master is not None:
            master.stop()
        else:
            [p.kill() for p in eval_processes if p.is_alive()]
        return all_performance / len(all_result), performances / len(all_result), performance_matrix / len(
            all_result), all_result

    def evaluate(self, problem_instance: BaseProblem, seed: int = 1088):
        solver_list = [self.solver_class(config) for config in self.config_list]
        return np.max([solver.optimize(problem_instance=problem_instance, seed=seed)[1] for solver in solver_list])
