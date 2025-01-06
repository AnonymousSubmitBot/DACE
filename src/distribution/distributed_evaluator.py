import _thread
import json
import logging
import os
import pickle
import signal
import time
from json import JSONDecodeError
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue, Manager, Lock
from pathlib import Path

import logzero
import requests
import urllib3
from flask import Flask, request

from src.distribution.util import get_host, md5_encode, exit_simulator
from src.types_ import *

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)


class DistributedEvaluator(Process):
    def __init__(self, port=2333, max_parallel_num=40, server_host="172.18.36.128", eval_method=None):
        super().__init__()
        if eval_method is None:
            raise ValueError("eval_method ERROR")
        self.logger = logzero.setup_logger(
            logfile=str(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/evaluator.log")),
            name="DistributedEvaluator Log", level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
            maxBytes=int(1e7))
        self.server_host = server_host
        self.port = port
        self.app = Flask(__name__)
        self.app.add_url_rule('/eval_new_task', view_func=self.eval_new_task, methods=['POST'])
        self.app.add_url_rule('/check_availability', view_func=self.check_availability, methods=['POST'])
        self.app.add_url_rule('/check_reachability', view_func=self.check_reachability, methods=['POST'])
        self.app.add_url_rule('/terminate', view_func=self.terminate_process, methods=['POST'])
        file_handler = RotatingFileHandler(
            Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/evaluator_flask.log"),
            maxBytes=int(1e7), backupCount=10)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.app.logger.addHandler(file_handler)
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.addHandler(file_handler)
        werkzeug_logger.setLevel(logging.DEBUG)
        self.set_middleware()
        self.manager = Manager()
        self.status = self.manager.dict()
        self.status_lock = Lock()
        self.max_parallel_num = max_parallel_num
        with self.status_lock:
            self.status["registered"] = False
        self.up_queue = Queue(self.max_parallel_num)
        self.score_queue = Queue()
        self.eval_queue = Queue()
        self.eval_processes = []
        self.eval_method = eval_method
        self.get_max_parallel_num()
        for p_num in range(self.max_parallel_num):
            p = Process(target=self.eval_method, args=(self.up_queue, self.score_queue, self.eval_queue, True))
            p.start()
            self.eval_processes.append(p)
        self.return_score_process = Process(target=self.return_score_result, args=())
        self.return_eval_process = Process(target=self.return_eval_result, args=())
        self.return_score_process.start()
        self.return_eval_process.start()

    def run(self):
        self.check_reachability_process = Process(target=self.check_master_reachability)
        self.check_reachability_process.start()
        self.app.run(debug=False, port=self.port, host='0.0.0.0', ssl_context='adhoc', threaded=True)

    def get_max_parallel_num(self):
        response_code = 500
        while response_code > 299 or response_code < 200:
            try:
                with self.status_lock:
                    response = requests.post(url="https://{}:1088/get_eval_capacity".format(self.server_host),
                                             data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM",
                                                   "capacity": self.max_parallel_num}, verify=False)
                response_code = response.status_code
                self.logger.info(
                    "Send Register Request to server {} with code {}".format(self.server_host, response_code))
            except requests.exceptions.ConnectionError as e:
                print("GET Parallel Num ERROR, Retry", end="\r")
                pass
            time.sleep(2)
        self.max_parallel_num = int(json.loads(response.content.decode("utf"))["capacity"])
        print("Parallel Num is {}".format(self.max_parallel_num))

    def register(self):
        response_code = 500
        host_result = get_host()
        if host_result[0]:
            while response_code > 299 or response_code < 200:
                try:
                    with self.status_lock:
                        response = requests.post(url="https://{}:1088/add_evaluator".format(self.server_host),
                                                 data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM",
                                                       "host": host_result[2], "port": self.port,
                                                       "capacity": self.max_parallel_num}, verify=False)
                    response_code = response.status_code
                    self.logger.info(
                        "Send Register Request to server {} with code {}".format(self.server_host, response_code))
                except requests.exceptions.ConnectionError as e:
                    print("Register ERROR, Retry", end="\r")
                    pass
                time.sleep(2)
            self.logger.info("Register to the server {}".format(self.server_host))
            with self.status_lock:
                self.status["registered"] = True
            print("\nRegistered")
        else:
            raise ValueError("Get Network Host Failed")

    def set_middleware(self):
        @self.app.before_request
        def check_verification():
            try:
                data = request.form.to_dict()
                check_val = md5_encode(md5_encode(md5_encode(md5_encode(md5_encode(data["check_val"])))))
                if check_val != "81600a92e8416bba7d9fada48e9402a4":
                    return {"success": False, "msg": "ERROR"}
            except:
                return {"success": False, "msg": "ERROR"}

    def eval_new_task(self):
        task_data = pickle.loads(request.files['task_data'].stream.read())
        self.up_queue.put(task_data)
        return {"success": True, "msg": "Submit Task Successfully"}

    def check_availability(self):
        available = self.up_queue.qsize() < 1
        return {"success": True, "msg": available}

    def terminate_process(self):
        [p.kill() for p in self.eval_processes]
        _thread.start_new_thread(exit_simulator, ())
        return "OK"

    def restart(self):
        [p.kill() for p in self.eval_processes]
        self.return_score_process.kill()
        self.return_eval_process.kill()
        time.sleep(3)
        parent_pid = os.getppid()
        os.kill(parent_pid, signal.SIGKILL)

    def return_eval_result(self):
        while True:
            result_data = self.eval_queue.get()
            response_code = 500
            while response_code > 299 or response_code < 200:
                try:
                    response = requests.post(url="https://{}:1088/post_eval_result".format(self.server_host),
                                             data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"},
                                             files={"result": pickle.dumps(result_data)}, verify=False)
                    response_code = response.status_code
                except requests.exceptions.ConnectionError as e:
                    pass
            # self.logger.info('Return a eval result to {}, with code {}'.format(self.server_host, response_code))

    def check_reachability(self):
        return {"success": True, "msg": True}

    def return_score_result(self):
        while True:
            result_data = self.score_queue.get()
            response_code = 500
            while response_code > 299 or response_code < 200:
                try:
                    response = requests.post(url="https://{}:1088/post_score_result".format(self.server_host),
                                             data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"},
                                             files={"result": pickle.dumps(result_data)}, verify=False)
                    response_code = response.status_code
                except requests.exceptions.ConnectionError as e:
                    pass
            # self.logger.info('Return a score result to {}, with code {}'.format(self.server_host, response_code))

    def check_master_reachability(self):
        self.logger.info("START Evaluator Reachability Check")
        while True:
            time.sleep(2)
            try_time = 0
            with self.status_lock:
                registered = self.status["registered"]
            if registered:
                while True:
                    print("CHECK REACHABILITY", end="\r")
                    try_time += 1
                    if try_time > 3:
                        self.logger.info("Lose Master")
                        with self.status_lock:
                            self.status["registered"] = False
                        self.restart()
                        break
                    try:
                        response = requests.post(url="https://{}:1088/check_reachability".format(self.server_host),
                                                 data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"}, verify=False,
                                                 timeout=3)
                        response_code = response.status_code
                    except requests.exceptions.ConnectionError as e:
                        continue
                    if response_code > 299 or response_code < 200:
                        continue
                    if json.loads(response.content.decode("utf"))["msg"] == True:
                        break



if __name__ == '__main__':
    a = "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"
    print(md5_encode(md5_encode(md5_encode(md5_encode(md5_encode(a))))))
    print(md5_encode(md5_encode(md5_encode(md5_encode(md5_encode(a))))))
    print(md5_encode(md5_encode(md5_encode(md5_encode(md5_encode(a))))))
    print(md5_encode(md5_encode(md5_encode(md5_encode(md5_encode(a))))))
