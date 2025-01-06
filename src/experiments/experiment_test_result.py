import os
import pickle
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ranksums
from scipy.stats import wilcoxon

from src.experiments.experiment_problem import problem_domains
from src.types_ import *


def keep_best_norm(result, min_value, max_value):
    new_result = [result[0]]
    for index in range(1, len(result)):
        if result[index] > -1e15:
            new_result.append(max(new_result[-1], result[index]))
        else:
            break
    return (np.array(new_result) - min_value) / (max_value - min_value)


def normalize_aggregated_result(domain, train_dim, load_pickle_dir, data_dir, max_eval=4, solver_num=4):
    total_result = pickle.load(open(Path(load_pickle_dir, "{}_{}_total_result.pkl".format(domain, train_dim)), "rb"))
    ref_scores = {
        dim: [np.load(Path(data_dir, "problem_instance/test", f"{domain}_{dim}_{index}", "y.npy")) for index in
              range(problem_domains[domain]['test_num'])] for dim in problem_domains[domain]['test_dim']}
    aggregated_step_data = {}
    for dim in problem_domains[domain]['test_dim']:
        if "CEPS_logs" in str(load_pickle_dir) or "DACE_logs" in str(load_pickle_dir):
            pap_titles = [f'initial_{dim}'] + [f'{index}th_PAP_{dim}' for index in range(max_eval)]
            aggregated_step_data[dim] = {pap_title: [] for pap_title in pap_titles}
            for problem_index in range(problem_domains[domain]['test_num']):
                min_value, max_value = np.min(ref_scores[dim][problem_index]), np.max(ref_scores[dim][problem_index])
                [aggregated_step_data[dim][pap_title].append([np.max(np.array(
                    [keep_best_norm(single_eval['step_history_matrix'][solver_index][problem_index], min_value,
                                    max_value)
                     for solver_index in range(solver_num)]), axis=0) for single_eval in total_result[dim][pap_title]])
                    for pap_title in pap_titles]
        else:
            aggregated_step_data[dim] = {"final_pap": []}
            for problem_index in range(problem_domains[domain]['test_num']):
                min_value, max_value = np.min(ref_scores[dim][problem_index]), np.max(ref_scores[dim][problem_index])
                aggregated_step_data[dim]["final_pap"].append([np.max(np.array(
                    [keep_best_norm(single_eval['step_history_matrix'][solver_index][problem_index], min_value,
                                    max_value)
                     for solver_index in range(solver_num)]), axis=0) for single_eval in total_result[dim]])

    pickle.dump(aggregated_step_data,
                open(Path(load_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim)), "wb"))


def plot_iter_pap_best_result(aggregated_step_datas, title):
    plt.clf()
    markers = ["o", "s"]
    linestyles = ["--", ":"]
    cap_sizes = [8, 5]
    elinewidths = [1.5, 2]
    colors = ["#FA8072", "#8A2BE2", "#7FFF00", "#FF4500", "#1A2C5B"]
    for id, method in enumerate(aggregated_step_datas.keys()):
        aggregated_step_data = aggregated_step_datas[method]
        pap_list = list(aggregated_step_data.keys())
        ins_result = [np.mean(aggregated_step_data[pap], axis=1)[:, -1] for pap in pap_list]
        x, mean, ci_95 = [], [], []
        for pap_index, pap in enumerate(pap_list):
            pap_name = "_".join(pap.split("_")[:-1])
            x.append(pap_name)
            average = np.mean(ins_result[pap_index])
            mean.append(average)
            sem = stats.sem(ins_result[pap_index])
            confidence = 0.95
            df = len(ins_result[pap_index]) - 1
            t_critical = stats.t.ppf((1 + confidence) / 2, df)
            margin_of_error = t_critical * sem
            ci_95.append(margin_of_error)
        x, mean, ci_95 = np.array(x), np.array(mean), np.array(ci_95)

        plt.errorbar(
            x, mean, ci_95, color=colors[id], marker=markers[id],
            linestyle=linestyles[id], label=method, markersize=6, capsize=cap_sizes[id],
            elinewidth=elinewidths[id]
        )
    plt.legend()
    plt.title(title)
    plt.show()
    # plt.savefig(f"../../figs/{title}.pdf", backend="pgf")

def statistic_wdl(dace_aggregated_data, compare_aggregated_datas):
    for method in compare_aggregated_datas.keys():
        compare_aggregated_data = compare_aggregated_datas[method]
        assert len(compare_aggregated_data) == len(dace_aggregated_data)
        wdl = [0, 0, 0]
        for index in range(len(compare_aggregated_data)):
            dace_data = np.array(dace_aggregated_data[index])[:, -1]
            compare_data = np.array(compare_aggregated_data[index])[:, -1]
            if ranksums(dace_data, compare_data, alternative="greater")[1] < 0.05:
                wdl[0] += 1
            elif ranksums(dace_data, compare_data, alternative="less")[1] < 0.05:
                wdl[2] += 1
            else:
                wdl[1] += 1
        print(method, f"{wdl[0]}↑\t{wdl[1]}→\t{wdl[2]}↓")


def statistic_total_mean_std_wdl(dace_aggregated_data, compare_aggregated_datas):
    for method in compare_aggregated_datas.keys():
        compare_aggregated_data = compare_aggregated_datas[method]
        assert len(compare_aggregated_data) == len(dace_aggregated_data)
        dace_data, compare_data = [], []
        for index in range(len(compare_aggregated_data)):
            dace_data.append(np.mean(np.array(dace_aggregated_data[index])[:, -1]))
            compare_data.append(np.mean(np.array(compare_aggregated_data[index])[:, -1]))
        if wilcoxon(dace_data, compare_data, alternative="greater")[1] < 0.05:
            wdl = "↓"
        elif wilcoxon(dace_data, compare_data, alternative="less")[1] < 0.05:
            wdl = "↑"
        else:
            wdl = "→"
        print(method,
              f"{np.mean(dace_data):.4f}±{np.std(dace_data):.4f} {np.mean(compare_data):.4f}±{np.std(compare_data):.4f} {wdl}")


def boxplot_total(aggregated_step_data_dict, domain):
    dims = list(aggregated_step_data_dict["dace"].keys())
    colors = ["#FA8072", "#8A2BE2", "#7FFF00", "#FF4500", "#1A2C5B"]
    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 6))
    for dim_index, dim in enumerate(dims):
        plot_dict = {
            "performance": [],
            "method": []
        }
        for method in aggregated_step_data_dict.keys():
            if method in ["dace", "ceps"]:
                pap_key = f"3th_PAP_{dim}"
            else:
                pap_key = "final_pap"
            if aggregated_step_data_dict[method] is None:
                break
            aggregated_step_data = aggregated_step_data_dict[method][dim][pap_key]
            performance = [np.mean([run_data[-1] for run_data in ins_data]) for ins_data in aggregated_step_data]
            plot_dict["performance"] += performance
            plot_dict["method"] += [method] * len(performance)
        df = pd.DataFrame(plot_dict)
        sns.boxplot(x='method', y='performance', data=df, ax=axes[dim_index],
                    whis=(2,98), width=.4/5*len(set(plot_dict["method"])), palette=colors)
        axes[dim_index].set_title(f"{domain} {dim}")
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"../../figs/boxplot_{domain}.pdf", backend="pgf")


def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    CEPS_pickle_dir = Path(file_dir, "../../logs/CEPS_logs/test_set_eval")
    DACE_pickle_dir = Path(file_dir, "../../logs/DACE_logs/test_set_eval")
    BRKGA_pickle_dir = Path(file_dir, "../../logs/Default_BRKGA_logs/test_set_eval")
    SMARTEST_pickle_dir = Path(file_dir, "../../logs/Default_SMARTEST_logs/test_set_eval")
    GLOBAL_pickle_dir = Path(file_dir, "../../logs/GLOBAL_logs/test_set_eval")
    PARHYDRA_pickle_dir = Path(file_dir, "../../logs/PARHYDRA_logs/test_set_eval")
    data_dir = Path(file_dir, "../../data")
    for domain in problem_domains.keys():
        train_dim = problem_domains[domain]['train_dim'][0]
        for load_pickle_dir in [CEPS_pickle_dir, DACE_pickle_dir, BRKGA_pickle_dir, GLOBAL_pickle_dir,
                                PARHYDRA_pickle_dir]:
            print(domain, load_pickle_dir)
            if not os.path.exists(Path(load_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim))):
                normalize_aggregated_result(domain, train_dim, load_pickle_dir, data_dir)
        ceps_step_data = pickle.load(
            open(Path(CEPS_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim)), "rb"))
        dace_step_data = pickle.load(
            open(Path(DACE_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim)), "rb"))
        brkga_step_data = pickle.load(
            open(Path(BRKGA_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim)), "rb"))
        global_step_data = pickle.load(
            open(Path(GLOBAL_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim)), "rb")
        )
        parhydra_step_data = pickle.load(
            open(Path(PARHYDRA_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim)), "rb")
        )
        if domain == "compiler_args_selection_problem":
            load_pickle_dir = SMARTEST_pickle_dir
            if not os.path.exists(Path(load_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim))):
                normalize_aggregated_result(domain, train_dim, load_pickle_dir, data_dir)
            smartest_step_data = pickle.load(
                open(Path(SMARTEST_pickle_dir, "aggregated_result_{}_{}.pkl".format(domain, train_dim)), "rb"))
        else:
            smartest_step_data = None
        plot_iter_pap_best_result({
            f"{dim}": dace_step_data[dim] for dim in ceps_step_data.keys()
        }, title=f"{domain}")
        boxplot_total({
            "dace": dace_step_data,
            "ceps": ceps_step_data,
            "parhydra": parhydra_step_data,
            "global": global_step_data,
            "brkga": brkga_step_data,
            "smartest": smartest_step_data
        }, domain=f"{domain}")
        for dim in ceps_step_data.keys():
            plot_iter_pap_best_result({
                "CEPS": ceps_step_data[dim],
                "DACE": dace_step_data[dim]
            }, title=f"{domain}_{dim}")
            statistic_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                          {"CEPS": ceps_step_data[dim][f"3th_PAP_{dim}"]})
            statistic_total_mean_std_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                                         {"CEPS": ceps_step_data[dim][f"3th_PAP_{dim}"]})
            statistic_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                          {"BRKGA": brkga_step_data[dim]["final_pap"]})
            statistic_total_mean_std_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                                         {"BRKGA": brkga_step_data[dim]["final_pap"]})
            statistic_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                          {"GLOBAL": global_step_data[dim]["final_pap"]})
            statistic_total_mean_std_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                                         {"GLOBAL": global_step_data[dim]["final_pap"]})
            statistic_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                          {"PARHYDRA": parhydra_step_data[dim]["final_pap"]})
            statistic_total_mean_std_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                                         {"PARHYDRA": parhydra_step_data[dim]["final_pap"]})
            if domain == "compiler_args_selection_problem":
                statistic_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                              {"SMARTEST": smartest_step_data[dim]["final_pap"]})
                statistic_total_mean_std_wdl(dace_step_data[dim][f"3th_PAP_{dim}"],
                                             {"SMARTEST": smartest_step_data[dim]["final_pap"]})


if __name__ == '__main__':
    main()
