from problem_domain import BaseProblem
from src.types_ import *


class BaseSolver(object):
    config_range: List[Tuple[Type[Union[float, int, bool]], Tuple, Union[float, int, bool]]] = []
    recommend_config:List[List[Union[float, int, bool]]] = []

    def __init__(self, config: List[Union[float, int, bool]], max_eval: int = 800, **kwargs):
        pass

    def optimize(self, problem_instance: BaseProblem, seed: int) -> Tuple[Union[NpArray, List[int], Tensor], float]:
        pass
