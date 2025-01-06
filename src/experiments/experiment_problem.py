import os
import pickle
import random
from multiprocessing import Process, Manager
from pathlib import Path

from src.problem_domain import BaseProblem, ContaminationProblem, ComInfluenceMaxProblem, CompilerArgsSelectionProblem
from src.types_ import *

problem_domains: Dict[str, Dict] = {
    "com_influence_max_problem": {
        "class": ComInfluenceMaxProblem,
        "train_dim": [80],
        "test_dim": [80, 100],
        "train_num": 5,
        "test_num": 50
    },
    "compiler_args_selection_problem": {
        "class": CompilerArgsSelectionProblem,
        "train_dim": [80],
        "test_dim": [80, 100],
        "train_num": 5,
        "test_num": 50
    },
    "contamination_problem": {
        "class": ContaminationProblem,
        "train_dim": [30],
        "test_dim": [30, 40],
        "train_num": 5,
        "test_num": 50
    }
}


def generate_problem_instance(ins_dir: str = '../data/problem_instance'):
    if not os.path.exists(Path(ins_dir, "train")):
        os.mkdir(Path(ins_dir, "train"))
    if not os.path.exists(Path(ins_dir, "test")):
        os.mkdir(Path(ins_dir, "test"))
    for problem_domain in problem_domains.keys():
        domain_setting = problem_domains[problem_domain]
        for train_dim in domain_setting["train_dim"]:
            for index in range(domain_setting["train_num"]):
                if not os.path.exists(Path(ins_dir, "train", "{}_{}_{}".format(problem_domain, train_dim, index))):
                    print("mkdir", str(Path(ins_dir, "train", "{}_{}_{}".format(problem_domain, train_dim, index))))
                    os.mkdir(Path(ins_dir, "train", "{}_{}_{}".format(problem_domain, train_dim, index)))
                    problem_instance = domain_setting["class"](dimension=train_dim, train=True)
                    pickle.dump(
                        problem_instance,
                        open(Path(ins_dir, "train", "{}_{}_{}".format(problem_domain, train_dim, index), "problem.pkl"),
                             "wb")
                    )
                    print("generate", "train_{}_{}_{}".format(problem_domain, train_dim, index))
        for test_dim in domain_setting["test_dim"]:
            for index in range(domain_setting["test_num"]):
                if not os.path.exists(Path(ins_dir, "test", "{}_{}_{}".format(problem_domain, test_dim, index))):
                    print("mkdir", str(Path(ins_dir, "test", "{}_{}_{}".format(problem_domain, test_dim, index))))
                    os.mkdir(Path(ins_dir, "test", "{}_{}_{}".format(problem_domain, test_dim, index)))
                    problem_instance = domain_setting["class"](dimension=test_dim, train=False)
                    pickle.dump(
                        problem_instance,
                        open(Path(ins_dir, "test", "{}_{}_{}".format(problem_domain, test_dim, index), "problem.pkl"),
                             "wb")
                    )
                    print("generate", "test_{}_{}_{}".format(problem_domain, test_dim, index))



def load_problem_instance(problem_dir: Path) -> BaseProblem:
    return pickle.load(open(Path(problem_dir, "problem.pkl"), "rb"))


def generate_data_from_problem_instance(problem: BaseProblem, sample_num=20000):
    solutions = set()
    for _ in range(sample_num):
        temp = tuple(random.randint(0, 1) for _ in range(problem.dimension))
        while temp in solutions:
            temp = tuple(random.randint(0, 1) for _ in range(problem.dimension))
        solutions.add(temp)
    x: NpArray = np.array(list(solutions), dtype=np.float32)
    y: NpArray = np.zeros([sample_num], dtype=np.float32)
    for i in range(sample_num):
        y[i] = problem.evaluate(solution=np.array(x[i], dtype=np.int32))
    return x, y


def evaluate_batch_instance(problem: BaseProblem, x_batch: NpArray, result_list):
    y_batch: NpArray = np.zeros([len(x_batch)], dtype=np.float32)
    for i in range(len(x_batch)):
        y_batch[i] = problem.evaluate(solution=np.array(x_batch[i], dtype=np.int32))
    result_list.append((x_batch, y_batch))


def generate_data_from_problem_instance_multi_process(problem: BaseProblem, sample_num=20000, p_num=-1):
    if p_num <= 0:
        p_num = 10 if isinstance(problem, ComInfluenceMaxProblem) else 200
    solutions = set()
    for _ in range(sample_num):
        temp = tuple(random.randint(0, 1) for _ in range(problem.dimension))
        while temp in solutions:
            temp = tuple(random.randint(0, 1) for _ in range(problem.dimension))
        solutions.add(temp)
    X: NpArray = np.array(list(solutions), dtype=np.float32)
    x_batches = np.array_split(X, p_num)
    manager = Manager()
    all_result = manager.list()
    process_list = []
    for x_batch in x_batches:
        p = Process(target=evaluate_batch_instance, args=(problem, x_batch, all_result))
        process_list.append(p)
        p.start()
    [p.join() for p in process_list]
    x = np.concatenate([result[0] for result in all_result])
    y = np.concatenate([result[1] for result in all_result])
    return x, y


def generate_problem_data(problem_dir: Path, sample_num=20000):
    if os.path.exists(Path(problem_dir, "x.npy")) and os.path.exists(Path(problem_dir, "y.npy")):
        return
    problem = load_problem_instance(problem_dir=problem_dir)
    x, y = generate_data_from_problem_instance_multi_process(problem, sample_num=sample_num)
    np.save(str(Path(problem_dir, "x.npy")), x)
    np.save(str(Path(problem_dir, "y.npy")), y)
    print("Generate New DATA for", str(problem_dir))


def load_problem_data(problem_dir):
    x = np.load(str(Path(problem_dir, "x.npy")))
    y = np.load(str(Path(problem_dir, "y.npy")))
    return x, y


def generate_only_solution(dimension: int = 30, sample_num: int = 10000):
    solutions = set()
    for _ in range(sample_num):
        temp = tuple(random.randint(0, 1) for _ in range(dimension))
        while temp in solutions:
            temp = tuple(random.randint(0, 1) for _ in range(dimension))
        solutions.add(temp)
    return np.array(list(solutions), dtype=np.float32)


def load_sample_indices(problem_dir: Path, sample_num: int = 1000) -> NpArray:
    indices_path = Path(problem_dir, "indices_{}.npy".format(sample_num))
    if os.path.exists(indices_path):
        return np.load(open(indices_path, 'rb'))
    else:
        x, y = load_problem_data(problem_dir=problem_dir)
        length = len(x)
        indices = np.random.choice(length, sample_num, replace=False)
        np.save(open(indices_path, "wb"), indices)
        return indices


if __name__ == '__main__':
    root_dir = str(Path(os.path.dirname(os.path.abspath(__file__)), "../../data/problem_instance"))
    generate_problem_instance(ins_dir=root_dir)
    for instance_path in os.listdir(Path(root_dir, "test")):
        generate_problem_data(Path(root_dir, "test", instance_path), 100000)
    for instance_path in os.listdir(Path(root_dir, "train")):
        generate_problem_data(Path(root_dir, "train", instance_path), 100000)
