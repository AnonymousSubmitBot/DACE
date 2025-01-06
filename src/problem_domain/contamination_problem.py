from src.problem_domain.base_problem import BaseProblem
from src.types_ import *


class ContaminationProblem(BaseProblem):
    """
    Contamination Control Problem with the simplest graph
    """

    def __init__(self, dimension: int = 10, train: bool = False, random_seed_pair=(None, None), **kwargs):
        self.dimension = dimension
        if train:
            self.lamda = 0.0001
        else:
            self.lamda = np.random.choice([0, 0.01])
        super().__init__(self.dimension, **kwargs)

        self.adjacency_mat = []
        self.random_seed_info = 'R'.join(
            [str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])

        self.n_simulations = 100
        self.init_alpha = 1.0
        self.init_beta = 30.0
        self.contam_alpha = 1.0
        self.contam_beta = 17.0 / 3.0
        self.restore_alpha = 1.0
        self.restore_beta = 3.0 / 7.0
        # In all evaluation, the same sampled values are used.
        self.init_Z, self.lambdas, self.gammas = self._generate_contamination_dynamics(random_seed_pair[0])

    def _generate_contamination_dynamics(self, random_seed=None):
        init_Z = np.random.RandomState(random_seed).beta(self.init_alpha, self.init_beta, size=(self.n_simulations,))
        lambdas = np.random.RandomState(random_seed).beta(self.contam_alpha, self.contam_beta,
                                                          size=(self.dimension, self.n_simulations))
        gammas = np.random.RandomState(random_seed).beta(self.restore_alpha, self.restore_beta,
                                                         size=(self.dimension, self.n_simulations))

        return init_Z, lambdas, gammas

    def _contamination(self, x, cost, init_Z, lambdas, gammas, U, epsilon):
        assert x.size == self.dimension

        rho = 1.0
        n_simulations = 100

        Z = np.zeros((x.size, n_simulations))
        Z[0] = lambdas[0] * (1.0 - x[0]) * (1.0 - init_Z) + (1.0 - gammas[0] * x[0]) * init_Z
        for i in range(1, self.dimension):
            Z[i] = lambdas[i] * (1.0 - x[i]) * (1.0 - Z[i - 1]) + (1.0 - gammas[i] * x[i]) * Z[i - 1]

        below_threshold = Z < U
        constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

        return np.sum(x * cost - rho * constraints)

    def evaluate(self, solution: Union[NpArray, List[int], Tensor]):
        if not isinstance(solution, Tensor):
            solution = torch.tensor(solution)
        assert solution.dim() == 1
        assert solution.numel() == self.dimension
        if solution.dim() == 2:
            solution = solution.squeeze(0)
        evaluation = self._contamination(x=(solution.cpu() if solution.is_cuda else solution).numpy(),
                                         cost=np.ones(solution.numel()),
                                         init_Z=self.init_Z, lambdas=self.lambdas, gammas=self.gammas, U=0.1,
                                         epsilon=0.05)
        evaluation += self.lamda * float(np.sum((solution.cpu() if solution.is_cuda else solution).numpy()))
        return -float(evaluation * solution.new_ones((1,)).float())


if __name__ == '__main__':
    co = ContaminationProblem(dimension=25, lamda=0.01, random_seed_pair=(10, 100))
    solutions = np.array([np.random.randint(2, size=25) for _ in range(10)])
    print([co.evaluate(s) for s in solutions])
