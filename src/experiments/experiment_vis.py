import gc
import os
import pickle
import random
from multiprocessing import Process, Manager
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.signal.windows import hamming
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
import yaml

from src.problem_domain import BaseProblem, ComInfluenceMaxProblem, SurrogateProblem
from src.experiments.experiment_problem import problem_domains
from src.surrogate import SurrogateVAE
from src.types_ import *
from surrogate import SurrogateVAENoReg

file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = Path(file_dir, f'../../logs/vis_logs')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(Path(log_dir, "solution_and_fitness"), exist_ok=True)


def generate_solutions(dim: int = 30, sample_num: int = 100000):
    solutions = set()
    for _ in range(sample_num):
        temp = tuple(random.randint(0, 1) for _ in range(dim))
        while temp in solutions:
            temp = tuple(random.randint(0, 1) for _ in range(dim))
        solutions.add(temp)
    return np.array(list(solutions), dtype=np.int32)


def get_solutions(dim: int = 30):
    if not os.path.exists(Path(log_dir, f'solution_and_fitness/{dim}_solutions.npy')):
        X = generate_solutions(dim=dim)
        print("Generate Solution for dim", dim)
        np.save(Path(log_dir, f'solution_and_fitness/{dim}_solutions.npy'), X)
    return np.load(Path(log_dir, f'solution_and_fitness/{dim}_solutions.npy'))


def evaluate_batch_instance(problem: BaseProblem, x_batch: NpArray, batch_index: int, result_dict):
    y_batch: NpArray = np.zeros([len(x_batch)], dtype=np.float32)
    for i in range(len(x_batch)):
        y_batch[i] = problem.evaluate(solution=np.array(x_batch[i], dtype=np.int32))
    result_dict[batch_index] = y_batch


def eval_solution_with_problem_instance_multi_process(problem: BaseProblem, p_num=-1):
    if p_num <= 0:
        p_num = 10 if isinstance(problem, ComInfluenceMaxProblem) else 200
    X = get_solutions(problem.dimension)
    x_batches = np.array_split(X, p_num)
    manager = Manager()
    result_dict = manager.dict()
    process_list = []
    for batch_index, x_batch in enumerate(x_batches):
        p = Process(target=evaluate_batch_instance, args=(problem, x_batch, batch_index, result_dict))
        process_list.append(p)
        p.start()
    [p.join() for p in process_list]
    y = np.concatenate([result_dict[batch_index] for batch_index in range(len(x_batches))])
    return y


def eval_surrogate_problem(problem: SurrogateProblem):
    X = get_solutions(problem.dimension)
    score = problem.vae.forward(torch.tensor(X, dtype=torch.float32).to(problem.device),
                                problem.ins_emb, train=False)[3]
    return score.detach().cpu().numpy()


def get_fitness(domain: str):
    p_num_dict = {
        "com_influence_max_problem": 100,
        "compiler_args_selection_problem": 300,
        "contamination_problem": 600
    }
    if not os.path.exists(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl')):
        fitness_dict = {}
    else:
        fitness_dict = pickle.load(open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "rb"))
    domain_info = problem_domains[domain]
    for ins_type in ["train", "test"]:
        for dim in domain_info[f"{ins_type}_dim"]:
            for ins_index in range(domain_info[f"{ins_type}_num"]):
                if f"{ins_type}_{dim}_{ins_index}" not in fitness_dict:
                    ins = pickle.load(open(Path(
                        file_dir, f'../../data/problem_instance/{ins_type}/{domain}_{dim}_{ins_index}/problem.pkl'
                    ), "rb"))
                    fitness_vec = eval_solution_with_problem_instance_multi_process(ins, p_num_dict[domain])
                    fitness_dict[f"{ins_type}_{dim}_{ins_index}"] = fitness_vec
                    pickle.dump(fitness_dict, open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "wb"))
                    print(f"Evaluate {domain}_{ins_type}_{dim}_{ins_index}")
    dim = domain_info[f"train_dim"][0]
    ceps_record = pickle.load(open(Path(log_dir, f'../CEPS_logs/{domain}_{dim}.pkl'), "rb"))
    if "ceps_problem_instances" not in fitness_dict:
        fitness_dict["ceps_problem_instances"] = []
    last_len = 5
    for update_index, update_record in enumerate(ceps_record["update_history"]):
        if len(fitness_dict["ceps_problem_instances"]) > update_index + 1:
            last_len = len(update_record["problem_pop"])
            continue
        elif len(fitness_dict["ceps_problem_instances"]) <= update_index:
            fitness_dict["ceps_problem_instances"].append([])
        if "problem_pop" in update_record.keys():
            problems = update_record["problem_pop"][:-last_len]
            for problem_index, problem_ins in enumerate(problems):
                if len(fitness_dict["ceps_problem_instances"][update_index]) <= problem_index:
                    fitness_vec = eval_solution_with_problem_instance_multi_process(problem_ins, p_num_dict[domain])
                    fitness_dict["ceps_problem_instances"][update_index].append(fitness_vec)
                    pickle.dump(fitness_dict, open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "wb"))
                    print(f"Evaluate {domain}_{ins_type}_{dim} CEPS ins, "
                          f"with {problem_index}th ins in {update_index}th update")
            last_len = len(update_record["problem_pop"])

    dace_record = pickle.load(open(Path(log_dir, f'../DACE_logs/{domain}_{dim}.pkl'), "rb"))
    log_path = Path(log_dir, "../surrogate_logs", "{}-{}".format(domain, dim))
    model_config = yaml.safe_load(open(Path(log_path, "HyperParam.yaml"), 'r'))
    model_dict = torch.load(str(Path(log_path, "best_model.pt")), map_location=torch.device("cpu"))
    if "dace_problem_initial_instances" not in fitness_dict:
        fitness_dict["dace_problem_initial_instances"] = []
    initial_problems = dace_record["initial_problems"]
    surrogate = SurrogateProblem(dimension=dim, domain=domain, vae_model=SurrogateVAE(**model_config['vae_params']),
                                 vae_weights=model_dict['vae_model'], ins_emb=torch.tensor(initial_problems[0]),
                                 gpu_index=0, reference_score=(0, 1))
    if len(fitness_dict["dace_problem_initial_instances"]) != len(initial_problems):
        fitness_dict["dace_problem_initial_instances"] = []
        for problem_index, initial_problem in enumerate(initial_problems):
            surrogate.update_ins_emb(torch.tensor(initial_problem), reference_score=(0, 1))
            fitness_vec = eval_surrogate_problem(surrogate)
            fitness_dict["dace_problem_initial_instances"].append(fitness_vec)
            pickle.dump(fitness_dict, open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "wb"))
            print(f"Evaluate {domain}_{ins_type}_{dim} DACE ins, with {problem_index}th ins in initial problems.")
    if "dace_problem_instances" not in fitness_dict:
        fitness_dict["dace_problem_instances"] = []
    last_len = 5
    for update_index, update_record in enumerate(dace_record["update_history"]):
        if len(fitness_dict["dace_problem_instances"]) > update_index + 1:
            last_len = len(update_record["problem_pop"])
            continue
        elif len(fitness_dict["dace_problem_instances"]) <= update_index:
            fitness_dict["dace_problem_instances"].append([])
        if "problem_pop" in update_record.keys():
            problems = update_record["problem_pop"][:-last_len]
            for problem_index, problem_ins_emb in enumerate(problems):
                if len(fitness_dict["dace_problem_instances"][update_index]) <= problem_index:
                    surrogate.update_ins_emb(torch.tensor(problem_ins_emb), reference_score=(0, 1))
                    fitness_vec = eval_surrogate_problem(surrogate)
                    fitness_dict["dace_problem_instances"][update_index].append(fitness_vec)
                    pickle.dump(fitness_dict, open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "wb"))
                    print(f"Evaluate {domain}_{ins_type}_{dim} DACE ins, "
                          f"with {problem_index}th ins in {update_index}th update")
            last_len = len(update_record["problem_pop"])

    dace_noreg_record = pickle.load(open(Path(log_dir, f'../DACENoReg_logs/{domain}_{dim}.pkl'), "rb"))
    log_path = Path(log_dir, "../surrogate_logs", "{}-{}".format(domain, dim))
    model_config = yaml.safe_load(open(Path(log_path, "HyperParam.yaml"), 'r'))
    model_dict = torch.load(str(Path(log_path, "best_model.pt")), map_location=torch.device("cpu"))
    vae_weights = model_dict['vae_model']
    weight_keys = list(vae_weights.keys())
    for key in weight_keys:
        if "scorer.weight_generator" in key.lower():
            vae_weights.pop(key)
    if "dace_noreg_problem_initial_instances" not in fitness_dict:
        fitness_dict["dace_noreg_problem_initial_instances"] = []
    initial_problems = dace_noreg_record["initial_problems"]
    surrogate = SurrogateProblem(dimension=dim, domain=domain,
                                 vae_model=SurrogateVAENoReg(**model_config['vae_params']),
                                 vae_weights=vae_weights, ins_emb=torch.tensor(initial_problems[0]),
                                 gpu_index=0, reference_score=(0, 1))
    if len(fitness_dict["dace_noreg_problem_initial_instances"]) != len(initial_problems):
        fitness_dict["dace_noreg_problem_initial_instances"] = []
        for problem_index, initial_problem in enumerate(initial_problems):
            surrogate.update_ins_emb(torch.tensor(initial_problem), reference_score=(0, 1))
            fitness_vec = eval_surrogate_problem(surrogate)
            fitness_dict["dace_noreg_problem_initial_instances"].append(fitness_vec)
            pickle.dump(fitness_dict, open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "wb"))
            print(f"Evaluate {domain}_{ins_type}_{dim} DACENoReg ins, with {problem_index}th ins in initial problems.")
    if "dace_noreg_problem_instances" not in fitness_dict:
        fitness_dict["dace_noreg_problem_instances"] = []
    last_len = 5
    for update_index, update_record in enumerate(dace_noreg_record["update_history"]):
        if len(fitness_dict["dace_noreg_problem_instances"]) > update_index + 1:
            last_len = len(update_record["problem_pop"])
            continue
        elif len(fitness_dict["dace_noreg_problem_instances"]) <= update_index:
            fitness_dict["dace_noreg_problem_instances"].append([])
        if "problem_pop" in update_record.keys():
            problems = update_record["problem_pop"][:-last_len]
            for problem_index, problem_ins_emb in enumerate(problems):
                if len(fitness_dict["dace_noreg_problem_instances"][update_index]) <= problem_index:
                    surrogate.update_ins_emb(torch.tensor(problem_ins_emb), reference_score=(0, 1))
                    fitness_vec = eval_surrogate_problem(surrogate)
                    fitness_dict["dace_noreg_problem_instances"][update_index].append(fitness_vec)
                    pickle.dump(fitness_dict, open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "wb"))
                    print(f"Evaluate {domain}_{ins_type}_{dim} DACENoReg ins, "
                          f"with {problem_index}th ins in {update_index}th update")
            last_len = len(update_record["problem_pop"])

    return fitness_dict


def solution_fitness_heatmap(solutions, fitness: NpArray):
    heat_value = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))
    pca = PCA(n_components=2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_pca = scaler.fit_transform(pca.fit_transform(solutions))
    num_grid = 50
    x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
    y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
    xi = np.linspace(x_min, x_max, num_grid)
    yi = np.linspace(y_min, y_max, num_grid)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata(X_pca, heat_value, (xi, yi), method='nearest')
    zi_nearest = griddata(X_pca, heat_value, (xi, yi), method='nearest')
    zi = np.where(np.isnan(zi), zi_nearest, zi)
    plt.figure(figsize=(10, 8))
    plt.imshow(zi, origin='lower', extent=(x_min, x_max, y_min, y_max),
               cmap='plasma', aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Heat Value', fontsize=12)
    plt.show()


def get_percent_value(solutions, fitness: NpArray, value_num: int = 20):
    fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))
    percent_generate = np.power([1 / (value_num) * i for i in range(value_num + 1)], 1 / 2) * 100
    percent_value = np.percentile(fitness, percent_generate, method='higher')
    return percent_value


def get_value_percent(solutions, fitness: NpArray, value_num: int = 20):
    fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))
    value_generate = np.power([1 / (value_num) * i for i in range(value_num + 1)], 1)
    np.sort(fitness)
    known_index = 0
    value_percent = []
    for rank, value in enumerate(value_generate):
        while fitness[known_index] < value:
            known_index += 1
        value_percent.append(known_index / len(fitness) * 100)
    return np.array(value_percent)


def statistic_dis_value(eval_indices, all_solutions, all_fitness, return_dict):
    for eval_index in eval_indices:
        sub_hamming_dict = {i: [] for i in range(len(all_solutions[0]))}
        hamming_vec = np.sum(np.abs(all_solutions[eval_index] - all_solutions), axis=1)
        fit_vec = np.abs(all_fitness[eval_index] - all_fitness)
        for i in range(len(hamming_vec)):
            sub_hamming_dict[hamming_vec[i]].append(fit_vec[i])
        return_dict[eval_index] = sub_hamming_dict
        # del hamming_vec, fit_vec, sub_hamming_dict
    # del all_solutions
    # gc.collect()


def distance_feature(solutions, fitness: NpArray, value_num: int = 15):
    np.random.seed(1088)
    fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))
    sample_indices = np.random.choice(list(range(len(solutions))), 100, replace=False)
    hamming_dict = {i: [] for i in range(len(solutions[0]))}
    p_list = []
    manager = Manager()
    result_dict = manager.dict()
    indices_batches = np.array_split(sample_indices, 20)
    for indices in indices_batches:
        p = Process(target=statistic_dis_value, args=(indices, solutions, fitness, result_dict))
        p.start()
        p_list.append(p)
    [p.join() for p in p_list]
    for index in sample_indices:
        temp_dict = result_dict[index]
        for key in temp_dict.keys():
            hamming_dict[key] += temp_dict[key]
    distances = list([i for i in hamming_dict.keys() if len(hamming_dict[i]) > 0])
    distances.sort()
    percent_generate = np.power([1 / (value_num) * i for i in range(value_num + 1)], 1) * 100
    selected_distance = np.percentile(distances, percent_generate, method='lower')
    # selected_distance = distances[:value_num]
    result = np.array([np.mean(hamming_dict[distance]) for distance in selected_distance] +
                      [np.std(hamming_dict[distance]) for distance in selected_distance])
    del hamming_dict
    gc.collect()
    return result


def get_hamming_matrix(dim: int = 80):
    def get_hamming_vec(a_index, all_solution, result):
        for index in a_index:
            result[index] = np.sum(np.abs(all_solution[index] - all_solution), axis=1)

    if not os.path.exists(Path(log_dir, f'solution_and_fitness/{dim}_hamming_matrix.npy')):
        solutions = get_solutions(dim)
        shape = solutions.shape
        hamming = np.zeros((shape[0], shape[0]))
        all_index = list(range(shape[0]))
        all_index_batch = np.array_split(all_index, 512)
        manager = Manager()
        result_dict = manager.dict()
        p_list = []
        for index_batch in all_index_batch:
            p = Process(target=get_hamming_vec, args=(index_batch, solutions, result_dict))
            p.start()
        [p.join() for p in p_list]

        for index in all_index:
            hamming[index] = result_dict[index]
        print("Generate Hamming Matrix for dim", dim)
        np.save(Path(log_dir, f'solution_and_fitness/{dim}_hamming_matrix.npy'), hamming)
    return np.load(Path(log_dir, f'solution_and_fitness/{dim}_hamming_matrix.npy'))


def union_value_and_percent(solutions, fitness: NpArray, value_num: int = 100):
    a = get_value_percent(solutions, fitness, value_num)
    b = get_percent_value(solutions, fitness, value_num)
    return np.concatenate([a, b])



def show_solution_distribution(feature_function=get_percent_value, reduction=None):
    def get_diff_result(func, args, return_list):
        result = func(*args)
        return_list.append(result)

    save_file = Path(log_dir, f"{feature_function.__name__}_{reduction.__class__.__name__}.pkl")
    if not os.path.exists(save_file):
        domain_feature = {}
        all_vec = []
        manager = Manager()
        for domain in problem_domains.keys():
            p_list = []
            if not os.path.exists(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl')):
                get_fitness(domain)
            fitness_dict = pickle.load(open(Path(log_dir, f'solution_and_fitness/{domain}_fitness.pkl'), "rb"))
            domain_info = problem_domains[domain]
            train_ins, test_ins, ceps_gen, dace_gen, dace_noreg = manager.list(), manager.list(), manager.list(), manager.list(), manager.list()
            for ins_type in ["train", "test"]:
                for dim in domain_info[f"{ins_type}_dim"]:
                    solution = get_solutions(dim)
                    for ins_index in range(domain_info[f"{ins_type}_num"]):
                        if ins_type == "train":
                            p = Process(target=get_diff_result,
                                        args=(feature_function,
                                              (solution, fitness_dict[f"{ins_type}_{dim}_{ins_index}"]),
                                              train_ins))
                            p.start()
                            p_list.append(p)
                            # train_ins.append(feature_function(solution, fitness_dict[f"{ins_type}_{dim}_{ins_index}"]))
                        else:
                            p = Process(target=get_diff_result,
                                        args=(feature_function,
                                              (solution, fitness_dict[f"{ins_type}_{dim}_{ins_index}"]),
                                              test_ins))
                            p.start()
                            p_list.append(p)
                            # test_ins.append(feature_function(solution, fitness_dict[f"{ins_type}_{dim}_{ins_index}"]))
            solution = get_solutions(domain_info[f"train_dim"][0])
            for update_index, problem_list in enumerate(fitness_dict["ceps_problem_instances"]):
                for fitness_vec in problem_list:
                    p = Process(target=get_diff_result, args=(feature_function, (solution, fitness_vec), ceps_gen))
                    p.start()
                    p_list.append(p)
            for fitness_vec in fitness_dict["dace_problem_initial_instances"]:
                p = Process(target=get_diff_result, args=(feature_function, (solution, fitness_vec), dace_gen))
                p.start()
                p_list.append(p)
            for update_index, problem_list in enumerate(fitness_dict["dace_problem_instances"]):
                for fitness_vec in problem_list:
                    p = Process(target=get_diff_result, args=(feature_function, (solution, fitness_vec), dace_gen))
                    p.start()
                    p_list.append(p)
            for fitness_vec in fitness_dict["dace_noreg_problem_initial_instances"]:
                p = Process(target=get_diff_result, args=(feature_function, (solution, fitness_vec), dace_noreg))
                p.start()
                p_list.append(p)
            for update_index, problem_list in enumerate(fitness_dict["dace_noreg_problem_instances"]):
                for fitness_vec in problem_list:
                    p = Process(target=get_diff_result, args=(feature_function, (solution, fitness_vec), dace_noreg))
                    p.start()
                    p_list.append(p)
            [p.join() for p in p_list]
            del solution
            train_ins_list, test_ins_list, ceps_gen_list, dace_gen_list, dace_noreg_list = list(train_ins), list(
                test_ins), list(ceps_gen), list(dace_gen), list(dace_noreg)
            domain_feature[domain] = (train_ins_list, test_ins_list, ceps_gen_list, dace_gen_list, dace_noreg_list)
            all_vec += train_ins_list + test_ins_list + ceps_gen_list + dace_gen_list + dace_noreg_list
            del train_ins, test_ins, ceps_gen, dace_gen, dace_noreg
            del train_ins_list, test_ins_list, ceps_gen_list, dace_gen_list, dace_noreg_list
            gc.collect()
        pickle.dump((domain_feature, all_vec), open(save_file, "wb"))
    domain_feature, all_vec = pickle.load(open(save_file, "rb"))
    scaler = MinMaxScaler(feature_range=(0, 1))
    trans_value = scaler.fit_transform(reduction.fit_transform(np.array(all_vec)))
    trans_dict = {}
    for index in range(len(all_vec)):
        trans_dict["_".join(map(str, all_vec[index]))] = trans_value[index]
    plt.figure(figsize=(10, 8))
    domain_name = ["CIM", "CA", "CON"]
    colors = ["#FA8072", "#7FFF00", "#8A2BE2", "#FF4500", "#1A2C5B"]
    labels = ["Train", "Test", "CEPS", "DACE", "NoReg"]
    markers = ["v", "s", "P", "h", "X"]
    zorders = [99, 0, 99, 99, 99]
    sizes = [50, 50, 50, 80, 40]
    for domain_index, domain in enumerate(problem_domains.keys()):
        for vec_index, vectors in enumerate(domain_feature[domain]):
            if len(vectors) == 0:
                continue
            coord = np.array([trans_dict["_".join(map(str, vec))] for vec in vectors])
            plt.scatter(coord[:, 0], coord[:, 1], color=colors[domain_index], marker=markers[domain_index],
                        label=f"{domain_name[domain_index]}",
                        zorder=zorders[0], s=[sizes[domain_index] for _ in coord])
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # plt.savefig("../../figs/total_vis.pdf", backend='pgf')

    markers = ["v", ".", "*", "h", "x"]
    sizes = [120, 150, 120, 180, 100]
    for domain_index, domain in enumerate(problem_domains.keys()):
        plt.figure(figsize=(10, 8))
        for vec_index, vectors in enumerate(domain_feature[domain]):
            if len(vectors) == 0:
                continue
            coord = np.array([trans_dict["_".join(map(str, vec))] for vec in vectors])
            plt.scatter(coord[:, 0], coord[:, 1], color=colors[vec_index], marker=markers[vec_index],
                        label=f"{domain_name[domain_index]}_{labels[vec_index]}",
                        s=[sizes[vec_index] for _ in coord]
                        )
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.show()
        # plt.savefig(f"../../figs/{domain}_vis.pdf", backend='pgf')


if __name__ == '__main__':
    get_fitness("com_influence_max_problem")
    get_fitness("compiler_args_selection_problem")
    get_fitness("contamination_problem")
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, learning_rate='auto',
                init='pca', perplexity=30, n_iter=10000)
    show_solution_distribution(reduction=tsne, feature_function=distance_feature)

