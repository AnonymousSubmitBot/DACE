import argparse
import datetime
import os
import pickle
import time
from pathlib import Path
from multiprocessing import Queue

from src.experiments.experiment_problem import problem_domains
from src.pap import BasePAP
from src.solver import BRKGASolver
from src.distribution import DistributedMaster
from src.solver.smartest_solver import SMARTESTSolver


def eval_default_pap(solver, domain, train_dim, max_parallel_num=200, repeat_time=4, solver_num=4, max_eval=800,
                     distributed=False):
    assert solver in ["BRKGA", "SMARTEST"]
    solver_classes = {
        "BRKGA": BRKGASolver,
        "SMARTEST": SMARTESTSolver,
    }
    pap = BasePAP(
        solver_class=solver_classes[solver],
        solver_num=solver_num,
        max_eval=max_eval
    )
    file_dir = os.path.dirname(os.path.abspath(__file__))
    load_pickle_dir = Path(file_dir, f"../../logs/Default_{solver}_logs/test_set_eval/{domain}_{train_dim}")
    if not os.path.exists(load_pickle_dir):
        os.makedirs(load_pickle_dir, exist_ok=True)
    total_result = {}
    for test_dim in problem_domains[domain]["test_dim"]:
        pap.set_config_list(solver_classes[solver].recommend_config)
        problem_list = []
        for index in range(problem_domains[domain]["test_num"]):
            problem_list.append(
                pickle.load(open(Path(file_dir, "../../data/problem_instance/test/{}_{}_{}/problem.pkl".format(
                    domain, test_dim, index
                )), "rb"))
            )
        print(f"---Default {solver} PAP for {domain} with dim={test_dim}------",
              datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
        all_perf, perfs, perf_mat, all_result = pap.eval_parallel_multi_times(
            problem_list,
            max_parallel_num=max_parallel_num,
            eval_times=repeat_time,
            load_pickle_path=str(Path(load_pickle_dir, "initial_{}.pkl".format(test_dim))),
            distributed=distributed
        )
        total_result[test_dim] = all_result
        print(all_perf)
        print("\t".join(map(str, perfs)))
    pickle.dump(total_result, open(Path(load_pickle_dir, "../{}_{}_total_result.pkl".format(domain, train_dim)), "wb"))


def eval_single_update_pap(domain, train_dim, max_parallel_num=200, repeat_time=4, solver_num=4, max_eval=800,
                           method="GLOBAL", distributed=False):
    assert method in ["GLOBAL", "PARHYDRA"]
    pap = BasePAP(
        solver_class=BRKGASolver,
        solver_num=solver_num,
        max_eval=max_eval
    )
    file_dir = os.path.dirname(os.path.abspath(__file__))
    load_pickle_dir = Path(file_dir, f"../../logs/{method}_logs/test_set_eval/{domain}_{train_dim}")
    if not os.path.exists(load_pickle_dir):
        os.makedirs(load_pickle_dir, exist_ok=True)
    total_result = {}
    experiment_record = pickle.load(
        open(Path(file_dir, "../../logs/{}_logs/{}_{}.pkl".format(method, domain, train_dim)),
             "rb")
    )
    pap.set_config_list(experiment_record[-1])
    for test_dim in problem_domains[domain]["test_dim"]:
        problem_list = []
        for index in range(problem_domains[domain]["test_num"]):
            problem_list.append(
                pickle.load(open(Path(file_dir, "../../data/problem_instance/test/{}_{}_{}/problem.pkl".format(
                    domain, test_dim, index
                )), "rb"))
            )
        print(f"---Default {method} PAP for {domain} with dim={test_dim}------",
              datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
        all_perf, perfs, perf_mat, all_result = pap.eval_parallel_multi_times(
            problem_list,
            max_parallel_num=max_parallel_num,
            eval_times=repeat_time,
            load_pickle_path=str(Path(load_pickle_dir, "initial_{}.pkl".format(test_dim))),
            distributed=distributed
        )
        total_result[test_dim] = all_result
        print(all_perf)
        print("\t".join(map(str, perfs)))
    pickle.dump(total_result, open(Path(load_pickle_dir, "../{}_{}_total_result.pkl".format(domain, train_dim)), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat_time', type=int, default=20, help='Repeat Times for PAP')
    parser.add_argument('--solver_num', type=int, default=4, help='Num of Solvers in PAP')
    parser.add_argument('--solver_max_eval', type=int, default=800, help='Max Eval Times for each Member Optimizer')
    args = parser.parse_args()
    for problem_domain in problem_domains.keys():
        if problem_domain == "com_influence_max_problem":
            max_parallel_num = 96
        elif problem_domain == "compiler_args_selection_problem":
            max_parallel_num = 250
        else:
            max_parallel_num = 600
        for problem_dim in problem_domains[problem_domain]["train_dim"]:
            eval_default_pap(solver="BRKGA", domain=problem_domain, train_dim=problem_dim,
                             max_parallel_num=max_parallel_num, repeat_time=args.repeat_time,
                             solver_num=args.solver_num, max_eval=args.solver_max_eval)
            eval_single_update_pap(domain=problem_domain, train_dim=problem_dim, max_parallel_num=max_parallel_num,
                                   repeat_time=args.repeat_time, solver_num=args.solver_num, method="GLOBAL",
                                   max_eval=args.solver_max_eval)
            eval_single_update_pap(domain=problem_domain, train_dim=problem_dim, max_parallel_num=max_parallel_num,
                                   repeat_time=args.repeat_time, solver_num=args.solver_num, method="PARHYDRA",
                                   max_eval=args.solver_max_eval)

    problem_domain = "compiler_args_selection_problem"
    max_parallel_num = 200
    for problem_dim in problem_domains[problem_domain]["train_dim"]:
        eval_default_pap(solver="SMARTEST", domain=problem_domain, train_dim=problem_dim,
                         max_parallel_num=max_parallel_num, repeat_time=args.repeat_time,
                         solver_num=args.solver_num, max_eval=args.solver_max_eval)
