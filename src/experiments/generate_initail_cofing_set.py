import os
import pickle
from pathlib import Path

from src.pap import BasePAP
from src.solver import BRKGASolver
from src.types_ import *

if __name__ == '__main__':
    config_nums = 50
    pap = BasePAP(solver_class=BRKGASolver, solver_num=4, max_eval=800)
    c: List[List[Union[float, int, bool]]] = [pap.sample_config() for _ in range(config_nums)]
    pickle.dump(c, open(
        Path(os.path.dirname(os.path.abspath(__file__)),
             '../../logs/initial_config_sets/brgka_configs_{}.pkl'.format(config_nums))
        , 'wb'))
