import _thread
import json
import logging
import os
import pickle
import time
from logging.handlers import RotatingFileHandler
from multiprocessing import Manager, Process, Queue, Lock
from pathlib import Path

import logzero
import requests
import urllib3
from flask import Flask, request

from src.distribution.util import md5_encode, exit_simulator, random_str
from src.types_ import *

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)


class DistributedMaster(Process):
    def __init__(self, port: int = 1088, up_queue: Queue = None, score_queue: Queue = None, eval_queue: Queue = None,
                 evaluator_capacity: int = None):
        super().__init__()
        self.logger = logzero.setup_logger(
            logfile=str(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/master.log")),
            name="DistributedMaster Log", level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
            maxBytes=int(1e7))
        self.port = port
        self.up_queue = up_queue
        self.score_queue = score_queue
        self.eval_queue = eval_queue
        self.manager = Manager()
        self.evaluator_dict = self.manager.dict()
        self.task_ongoing = self.manager.dict()
        self.task_lock = Lock()
        self.evaluator_lock = Lock()
        self.evaluator_capacity = evaluator_capacity
        self.app = Flask(__name__)
        self.app.add_url_rule('/add_evaluator', view_func=self.add_evaluator, methods=['POST'])
        self.app.add_url_rule('/get_eval_capacity', view_func=self.get_eval_capacity, methods=['POST'])
        self.app.add_url_rule('/post_eval_result', view_func=self.post_eval_result, methods=['POST'])
        self.app.add_url_rule('/post_score_result', view_func=self.post_score_result, methods=['POST'])
        self.app.add_url_rule('/check_reachability', view_func=self.check_reachability, methods=['POST'])
        file_handler = RotatingFileHandler(
            Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/master_flask.log"),
            maxBytes=int(1e7), backupCount=10)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.app.logger.addHandler(file_handler)
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.addHandler(file_handler)
        werkzeug_logger.setLevel(logging.DEBUG)

        self.set_middleware()
        self.distribute_process = Process(target=self.distribute_eval, args=(self.evaluator_dict, self.task_ongoing,))
        self.check_reachability_process = Process(target=self.check_evaluator_reachability,
                                                  args=(self.evaluator_dict, self.task_ongoing,))
        self.distribute_process.start()
        self.check_reachability_process.start()

    def run(self):
        self.app.run(debug=False, port=self.port, host='0.0.0.0', ssl_context='adhoc', threaded=True)

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

    def add_evaluator(self):
        data = request.form.to_dict()
        host = data["host"]
        port = data["port"]
        capacity = float(data["capacity"])
        with self.evaluator_lock:
            self.evaluator_dict["{}:{}".format(host, port)] = capacity
        self.logger.info("Add an Evaluator for host {} and port {} with capacity {}".format(host, port, capacity))
        self.logger.info("The Cluster Capacity is {}".format(self.get_total_capacity()))
        return_dict = {"success": True, "msg": "Add an Evaluator to the master"}
        return return_dict

    def get_eval_capacity(self):
        data = request.form.to_dict()
        capacity = float(data["capacity"])
        if self.evaluator_capacity is not None:
            capacity = self.evaluator_capacity
        return {"success": True, "capacity": capacity}

    def get_total_capacity(self) -> float:
        with self.evaluator_lock:
            total_capacity = sum([self.evaluator_dict[key] for key in self.evaluator_dict.keys()])
        return total_capacity

    def terminate_process(self):
        _thread.start_new_thread(exit_simulator, ())
        return "OK"

    def post_eval_result(self):
        result = pickle.loads(request.files['result'].stream.read())
        with self.task_lock:
            self.task_ongoing.pop(result[-1])
        self.eval_queue.put(result)
        # self.logger.info("Receive an Eval Result from {}".format(request.remote_addr))
        return {"success": True, "msg": "Receive an Eval Result from {}".format(request.remote_addr)}

    def post_score_result(self):
        result = pickle.loads(request.files['result'].stream.read())
        with self.task_lock:
            self.task_ongoing.pop(result[-1])
        self.score_queue.put(result)
        # self.logger.info("Receive an Score Result from {}".format(request.remote_addr))
        return {"success": True, "msg": "Receive an Score Result from {}".format(request.remote_addr)}

    def check_reachability(self):
        return {"success": True, "msg": True}

    def distribute_eval(self, evaluator_dict, task_ongoing):
        self.logger.info("START Distribute Evaluation Tasks")
        while True:
            # Get new Task
            task_data = self.up_queue.get()
            task_id = "{}_{}".format(str(int(time.time() * 1000)), random_str(32))
            task_data = task_data + (task_id,)
            response_code = 500
            send_host = ""
            while response_code > 299 or response_code < 200:
                # Use low overload Evaluator Priorly
                with self.evaluator_lock:
                    hosts = list(evaluator_dict.keys())
                    current_task_num = {host: 0 for host in hosts}
                    with self.task_lock:
                        current_tasks = dict(task_ongoing)
                        for current_id in current_tasks.keys():
                            if current_tasks[current_id]["send_host"] in current_task_num.keys():
                                current_task_num[current_tasks[current_id]["send_host"]] += 1
                    hosts.sort(key=lambda i: current_task_num[i] / evaluator_dict[i])
                for host in hosts:
                    try:
                        response = requests.post(url="https://{}/check_availability".format(host),
                                                 data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"}, verify=False)
                    except requests.exceptions.ConnectionError as e:
                        continue
                    if json.loads(response.content.decode("utf"))["msg"] == True:
                        try:
                            response = requests.post(url="https://{}/eval_new_task".format(host),
                                                     data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"},
                                                     files={"task_data": pickle.dumps(task_data)}, verify=False)
                            response_code = response.status_code
                            send_host = host
                        except requests.exceptions.ConnectionError as e:
                            continue
                        break
            with self.task_lock:
                task_ongoing[task_id] = {
                    "task_id": task_id,
                    "start_time": time.time(),
                    "task_data": task_data,
                    "send_host": send_host
                }
            # self.logger.info('Send a request to {}'.format(send_host))

    def close_evaluator(self):
        for host in self.evaluator_dict.keys():
            response_code = 500
            while response_code > 299 or response_code < 200:
                try:
                    response = requests.post(url="https://{}/terminate".format(host),
                                             data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"}, verify=False)
                    response_code = response.status_code
                except requests.exceptions.ConnectionError as e:
                    pass
            self.logger.info("Close the Evaluator {}".format(host))

    def stop(self):
        self.distribute_process.kill()
        self.check_reachability_process.kill()
        time.sleep(2)
        self.kill()

    def check_evaluator_reachability(self, evaluator_dict: Dict, task_ongoing: Dict):
        self.logger.info("START Evaluator Reachability Check")
        while True:
            time.sleep(2)
            for host in evaluator_dict.keys():
                try_time = 0
                while True:
                    try_time += 1
                    if try_time > 3:
                        with self.evaluator_lock:
                            evaluator_dict.pop(host)
                        self.logger.info("Delete the Invalid Evaluator {}".format(host))
                        self.logger.info("The Cluster Capacity is {}".format(self.get_total_capacity()))
                        with self.task_lock:
                            for task_id in task_ongoing.keys():
                                task = task_ongoing[task_id]
                                if task["send_host"] == host:
                                    task_data = task["task_data"][:-1]
                                    self.up_queue.put(task_data)
                                    task_ongoing.pop(task_id)
                                    self.logger.info("Restart the task with task_id {} on {}".format(task_id, host))
                        break
                    try:
                        response = requests.post(url="https://{}/check_reachability".format(host),
                                                 data={"check_val": "sTNas17^SC2jUdbIyJE3bb!XgqIIX+QM"}, verify=False,
                                                 timeout=3)
                        response_code = response.status_code
                    except requests.exceptions.ConnectionError as e:
                        continue
                    if response_code > 299 or response_code < 200:
                        continue
                    if json.loads(response.content.decode("utf"))["msg"] == True:
                        break
