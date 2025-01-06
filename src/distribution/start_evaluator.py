import argparse
import time
import sys
sys.path.append("/home/metaron/DACE_exp")
sys.path.append("/home/metaron/DACE_exp/src")

from src.distribution.distributed_evaluator import DistributedEvaluator
from src.pap.ceps import evaluate_problem_with_config as ceps_eval
from src.pap.base_pap import single_eval as base_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # music
    parser.add_argument('--pap', type=str, default="ceps", help='The type of the evaluator')
    parser.add_argument('--max_parallel_num', type=int, default=20, help='The max parallel num of the evaluator')
    parser.add_argument('--port', type=int, default=23333, help='The port of the evaluator')
    parser.add_argument('--server_host', type=str, default="172.18.36.128", help='The host of the server')
    args = parser.parse_args()
    if args.pap == "ceps":
        eval = ceps_eval
    elif args.pap == "base":
        eval = base_eval
    else:
        exit(0)
    while True:
        print("\nSTART Evaluator")
        evaluator = DistributedEvaluator(max_parallel_num=args.max_parallel_num, server_host=args.server_host,
                                         eval_method=eval, port=args.port)
        evaluator.start()
        time.sleep(1)
        evaluator.register()
        evaluator.join()
        print("\nEND Evaluator")
