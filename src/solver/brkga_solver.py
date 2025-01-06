import logging

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pytorch_lightning import seed_everything

from src.problem_domain import BaseProblem, CompilerArgsSelectionProblem
from src.solver.base_solver import BaseSolver
from src.types_ import *

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.get("hash") == b.get("hash")


class MyBitProblem(ElementwiseProblem):
    def __init__(self, eval_func: Callable[[NpArray], float], dimension: int, max_eval: int, low_better: bool = False):
        self.eval_func: Callable[[NpArray], float] = eval_func
        self.max_eval: int = max_eval
        self.low_better: bool = low_better
        self.eval_time: int = 0
        self.step_history = []
        self.quality_table: Dict[str, float] = {}
        super().__init__(n_var=dimension, n_obj=1, n_ieq_constr=0, xl=0, xu=1)

    def out_evaluate(self, x, out):
        self._evaluate(x, out)

    def _evaluate(self, x, out, *args, **kwargs):
        pheno = np.array([0 if value <= 0.5 else 1 for value in x], dtype=np.int32)
        pheno_str = "".join([str(i) for i in pheno])
        if self.eval_time >= self.max_eval:
            eval_value = -1e15 if not self.low_better else 1e15
        else:
            if pheno_str in self.quality_table.keys():
                eval_value = self.quality_table[pheno_str]
            else:
                eval_value = self.eval_func(pheno)
                self.quality_table[pheno_str] = eval_value
        self.eval_time += 1
        out["F"] = -eval_value if not self.low_better else eval_value
        out["pheno"] = pheno
        out["hash"] = hash(pheno_str)
        self.step_history.append(eval_value)


class BRKGASolver(BaseSolver):
    # Statistic Attribute
    config_range: List[Tuple[Type[Union[float, int, bool]], Tuple, Union[float, int, bool]]] = [
        (int, (1, 400), 200),
        (int, (1, 1000), 700),
        (int, (1, 200), 100),
        (float, (0, 1), 0.7),
        (bool, (True, False), False)
    ]
    # Parameters are referred from PyMOO and Biased random-key genetic algorithms for combinatorial optimization
    recommend_config = [
        [20, 70, 10, 0.7, False],
        [20, 70, 10, 0.7, True],
        [15, 75, 10, 0.7, False],
        [15, 75, 10, 0.7, True],
    ]

    def __init__(self,
                 config: List[Union[float, int, bool]] = None,
                 max_eval: int = 800,
                 **kwargs):
        super().__init__(config=config, max_eval=max_eval, **kwargs)
        # elements in config: [n_elites, n_offsprings, n_mutants, bias, eliminate_duplicates]
        self.max_eval = max_eval
        self.config = [200, 700, 100, 0.7, False] if config is None else config
        self.brkga = BRKGA(n_elites=self.config[0],
                           n_offsprings=self.config[1],
                           n_mutants=self.config[2],
                           bias=self.config[3],
                           sampling=BinaryRandomSampling(),
                           eliminate_duplicates=MyElementwiseDuplicateElimination() if self.config[4] else False
                           )

    def optimize(self, problem_instance: BaseProblem, seed: int = 1088, low_better: bool = False) -> Tuple[
        Union[NpArray, List[int], Tensor], float, List[float], Dict[str, float]]:
        problem = MyBitProblem(eval_func=problem_instance.evaluate, dimension=problem_instance.dimension,
                               max_eval=self.max_eval, low_better=low_better)
        seed_everything(seed, True)
        res = minimize(problem,
                       self.brkga,
                       termination=get_termination("n_eval", self.max_eval),
                       verbose=False,
                       seed=seed)
        best_x = res.opt.get("pheno")[0]
        best_y = -res.F[0] if not low_better else res.F[0]
        return best_x, best_y, problem.step_history, problem.quality_table
