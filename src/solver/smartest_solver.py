import random
from copy import deepcopy

from pytorch_lightning import seed_everything
from sklearn.ensemble import RandomForestRegressor

from src.problem_domain import BaseProblem
from src.solver.base_solver import BaseSolver
from src.types_ import *


class SMARTESTSolver(BaseSolver):
    config_range: List[Tuple[Type[Union[float, int, bool]], Tuple, Union[float, int, bool]]] = [
        (int, (1, 200), 100),
        (float, (0, 1), 0.8),
        (float, (0, 1), 0.2),
    ]
    recommend_config = [
        [100, 0.8, 0.1],
        [150, 0.8, 0.1],
        [100, 0.8, 0.2],
        [100, 0.5, 0.2],
    ]

    def __init__(self,
                 config: List[Union[float, int, bool]] = None,
                 max_eval: int = 800,
                 **kwargs,
                 ):
        super().__init__(config=config, max_eval=max_eval, **kwargs)
        self.config = config
        self.pop_size: int = self.config[0]
        self.crossover_prob: float = self.config[1]
        self.max_eval: int = max_eval
        self.elite_rate: float = self.config[2]
        self.dim: int = 0
        self.population: List[NpArray] = []
        self.population_quality: List[float] = []
        self.population_strs: Set[str] = set()
        self.eval_time = 0
        self.quality_table: Dict[str, float] = {}
        self.population_history: List[List[float]] = []
        self.step_history: List[float] = []

    def get_quality(self, problem_instance, solution: NpArray) -> float:
        solution_str = "".join([str(i) for i in solution])
        if self.eval_time > self.max_eval:
            return -1e15
        self.eval_time += 1
        if solution_str in self.quality_table:
            quality = self.quality_table[solution_str]
        else:
            quality = problem_instance.evaluate(solution)
            self.quality_table[solution_str] = quality
        self.step_history.append(quality)
        return quality

    def add_quality_record(self, solution: NpArray, quality: float):
        solution_str = "".join([str(i) for i in solution])
        self.quality_table[solution_str] = quality

    def add_solution_to_population(self, problem_instance, solution):
        solution_str = "".join([str(i) for i in solution])
        if solution_str not in self.population_strs:
            self.population.append(solution)
            self.population_strs.add(solution_str)
            quality = self.get_quality(problem_instance, solution)
            self.population_quality.append(quality)
            self.add_quality_record(solution, quality)
            return True
        else:
            return False

    def initial_population(self, problem_instance):
        self.population: List[NpArray] = []
        while len(self.population) < self.pop_size:
            self.add_solution_to_population(problem_instance, np.random.randint(2, size=self.dim))
        shuffle_index = list(range(self.pop_size))
        random.shuffle(shuffle_index)
        self.population = [self.population[index] for index in shuffle_index]
        self.population_quality = [self.population_quality[index] for index in shuffle_index]
        self.population_history.append(deepcopy(self.population_quality))

    def initial_surrogate(self):
        regr = RandomForestRegressor(max_depth=None, n_estimators=100, criterion="squared_error")
        x = np.array(self.population)
        y = np.array(self.population_quality)
        regr.fit(x, y)
        return regr

    def update_surrogate(self):
        self.surrogate = RandomForestRegressor(max_depth=None, n_estimators=100, criterion="squared_error")
        known_solutions = list(self.quality_table.keys())
        x = np.array([[int(bit) for bit in solution_str] for solution_str in known_solutions])
        y = np.array([self.quality_table[solution_str] for solution_str in known_solutions])
        self.surrogate.fit(x, y)

    def roulette_wheel_selection(self):
        weight = np.array(self.population_quality) / np.sum(self.population_quality)
        result = np.random.choice(self.pop_size, size=2, replace=False, p=weight)
        return result[0], result[1]

    def crossover(self):
        new_population = []
        r_num = self.pop_size - int(self.elite_rate * self.pop_size)
        while len(new_population) < r_num:
            i_1, i_2 = self.roulette_wheel_selection()
            n = np.random.randint(0, self.dim)
            p1, p2 = self.population[i_1].copy(), self.population[i_2].copy()
            if random.random() < self.crossover_prob:
                seg1, seg2 = p1[:n].copy(), p2[n:].copy()
                p1[n:], p2[:n] = seg2, seg1
            new_population.append(p1)
            new_population.append(p2)
        return np.array(new_population[:r_num])

    def local_search(self, solution):
        current_solution = solution.copy()
        while True:
            candidate_solutions: List = [deepcopy(current_solution) for _ in range(self.dim + 1)]
            for index in range(self.dim):
                candidate_solutions[index][index] = current_solution[index] ^ 1
            qualities = self.surrogate.predict(candidate_solutions)
            max_index = np.argmax(qualities)
            if qualities[max_index] == qualities[-1]:
                break
            else:
                current_solution = candidate_solutions[max_index].copy()
        return current_solution

    def get_pop_ab(self):
        elite_num = int(self.elite_rate * self.pop_size)
        sort_index = np.argsort(self.population_quality)[::-1]
        pop_a = [self.population[index] for index in sort_index[:elite_num]]
        pop_b = [self.population[index] for index in sort_index[elite_num:]]
        return pop_a, pop_b

    def merge_pop_b(self, crossover_result, local_search_result, pop_b):
        new_pop_b = []
        current_str_set = set(["".join([str(i) for i in solution]) for solution in self.population])
        for solution_set in [local_search_result, crossover_result, pop_b]:
            for solution in solution_set:
                if len(new_pop_b) >= len(pop_b):
                    break
                if "".join([str(i) for i in solution]) not in current_str_set:
                    new_pop_b.append(solution)
                    current_str_set.add("".join([str(i) for i in solution]))
        return new_pop_b[:len(pop_b)]

    def optimize(self, problem_instance: BaseProblem, seed: int) -> Tuple[
        Union[NpArray, List[int], Tensor], float, List[float], Dict[str, float]
    ]:
        seed_everything(seed, True)
        self.dim: int = problem_instance.dimension
        self.population: List[NpArray] = []
        self.population_quality: List[float] = []
        self.population_strs: Set[str] = set()
        self.eval_time = 0
        self.quality_table: Dict[str, float] = {}
        self.population_history: List[List[float]] = []
        self.step_history: List[float] = []
        self.initial_population(problem_instance)
        self.surrogate = self.initial_surrogate()
        while self.eval_time < self.max_eval:
            pop_a, pop_b = self.get_pop_ab()
            crossover_result = self.crossover()
            local_search_result = [self.local_search(solution=individual) for individual in crossover_result]
            self.population = pop_a
            self.population_strs = set(["".join([str(i) for i in solution]) for solution in self.population])
            self.population_quality = [self.quality_table["".join([str(i) for i in solution])] for solution in pop_a]
            new_pop_b = self.merge_pop_b(crossover_result, local_search_result, pop_b)
            for solution in new_pop_b:
                self.add_solution_to_population(problem_instance, solution)
                if self.eval_time >= self.max_eval:
                    break
            self.population_history.append(deepcopy(self.population_quality))
            self.update_surrogate()
        best_x_str = max(self.quality_table.keys(), key=lambda item: self.quality_table[item])
        best_x = [int(bit) for bit in best_x_str]
        best_y = self.quality_table[best_x_str]
        return best_x, best_y, self.step_history, self.quality_table
