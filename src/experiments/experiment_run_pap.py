import argparse
import datetime
import time
import os
import pickle
from distutils.util import strtobool
from pathlib import Path
from multiprocessing import Queue

from src.experiments.experiment_problem import problem_domains
from src.distribution import DistributedMaster
from src.pap import BasePAP
from src.solver import BRKGASolver


def main(domain, train_dim=80, max_parallel_num=200, repeat_time=4, solver_num=4, max_eval=800, method="CEPS",
         max_iter: int = 4, only_config_update: bool = False, distributed=False):
    pap = BasePAP(
        solver_class=BRKGASolver,
        solver_num=solver_num,
        max_eval=max_eval
    )
    file_dir = os.path.dirname(os.path.abspath(__file__))
    load_pickle_dir = Path(file_dir, "../../logs/{}_logs/test_set_eval/{}_{}".format(method, domain, train_dim))
    os.makedirs(load_pickle_dir, exist_ok=True)
    total_result = {}
    master = None
    if distributed:
        up_queue, eval_queue = Queue(), Queue()
        master = DistributedMaster(up_queue=up_queue,
                                   eval_queue=eval_queue,
                                   evaluator_capacity=max_parallel_num)
        master.start()
        time.sleep(5)
    else:
        up_queue, eval_queue = None, None
    for test_dim in problem_domains[domain]["test_dim"]:
        total_result[test_dim] = {}
        problem_list = []
        for index in range(problem_domains[domain]["test_num"]):
            problem_list.append(
                pickle.load(open(Path(file_dir, "../../data/problem_instance/test/{}_{}_{}/problem.pkl".format(
                    domain, test_dim, index
                )), "rb"))
            )
        experiment_record = pickle.load(
            open(Path(file_dir, "../../logs/{}_logs/{}_{}.pkl".format(method, domain, train_dim)),
                 "rb") if not only_config_update else open(
                Path(file_dir, "../../logs/{}_logs/{}_{}_only_update_config.pkl".format(method, domain, train_dim)),
                "rb")
        )
        for index in range(min(max_iter, len(experiment_record["update_history"])) - 1, -1, -1):
            pap.set_config_list(experiment_record["update_history"][index]["config_pop"])

            print("---{} {}th PAP for dim={}------".format(method, index, test_dim),
                  datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
            all_perf, perfs, perf_mat, all_result = pap.eval_parallel_multi_times(
                problem_list,
                max_parallel_num=max_parallel_num,
                eval_times=repeat_time, up_queue=up_queue, eval_queue=eval_queue,
                load_pickle_path=str(Path(load_pickle_dir, "{}th_PAP_{}.pkl".format(index, test_dim))),
                distributed=distributed
            )
            total_result[test_dim]["{}th_PAP_{}".format(index, test_dim)] = all_result
            print(all_perf)
            print("\t".join(map(str, perfs)))
            for performance_line in perf_mat:
                print("\t".join(map(str, performance_line)))
        pap.set_config_list(experiment_record["initial_configs"])

        print("---INITIAL PAP for dim={}------".format(test_dim),
              datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
        all_perf, perfs, perf_mat, all_result = pap.eval_parallel_multi_times(
            problem_list,
            max_parallel_num=max_parallel_num,
            eval_times=repeat_time, up_queue=up_queue, eval_queue=eval_queue,
            load_pickle_path=str(Path(load_pickle_dir, "initial_{}.pkl".format(test_dim))),
            distributed=distributed
        )
        total_result[test_dim]["initial_{}".format(test_dim)] = all_result
        print(all_perf)
        print("\t".join(map(str, perfs)))

        for performance_line in perf_mat:
            print("\t".join(map(str, performance_line)))
    pickle.dump(total_result, open(Path(load_pickle_dir, "../{}_{}_total_result.pkl".format(domain, train_dim)), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_domain', type=str, default="contamination_problem",
                        help='Optimization Problem Domain')
    parser.add_argument('--max_iter', type=int, default=4, help='Max Iteration Times for DACE/CEPS')
    parser.add_argument('--problem_dim', type=int, default=30, help='Optimization Problem Dimension')
    parser.add_argument('--repeat_time', type=int, default=20, help='Repeat Times for PAP')
    parser.add_argument('--solver_num', type=int, default=4, help='Num of Solvers in PAP')
    parser.add_argument('--solver_max_eval', type=int, default=800, help='Max Eval Times for each Member Optimizer')
    parser.add_argument('--max_parallel_num', type=int, default=200,
                        help='Max Parallel Processes for Evaluator on this machine')
    parser.add_argument('--method', type=str, default="DACE", help='PAP Construct Method')
    parser.add_argument('--only_config_update', type=strtobool, default=False,
                        help='Generate new Problem Instance or not')
    parser.add_argument('--distributed', type=strtobool, default=False, help='Eval PAP in Distributed Mode')
    args = parser.parse_args()

    main(domain=args.problem_domain, train_dim=args.problem_dim, max_parallel_num=args.max_parallel_num,
         repeat_time=args.repeat_time, solver_num=args.solver_num, max_eval=args.solver_max_eval,
         method=args.method, max_iter=args.max_iter, only_config_update=args.only_config_update,
         distributed=args.distributed
         )
