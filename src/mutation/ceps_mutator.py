from src.experiments.experiment_problem import problem_domains
from src.problem_domain import BaseProblem
from src.types_ import *


class CEPSMutator:
    def __init__(self, problem_dimension: int, problem_domain: str, popsize: int = 99):
        self.problem_dimension = problem_dimension
        self.problem_domain = problem_domain
        self.problem_class: Type[BaseProblem] = problem_domains[self.problem_domain]["class"]
        self.popsize = popsize

    def ask(self) -> List[BaseProblem]:
        return [self.problem_class(dimension=self.problem_dimension, train=True) for _ in range(self.popsize)]

    def tell(self, performance):
        pass
