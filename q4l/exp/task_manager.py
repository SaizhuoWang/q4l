import base64
import pickle
import time
import typing as tp
from typing import Any, Callable, Dict, Iterable, List, Tuple

import loky
import omegaconf
from bson.binary import Binary
from bson.objectid import ObjectId
from omegaconf import OmegaConf

from ..config import ExperimentConfig, GlobalConfig
from ..utils.log import get_logger
from .db_backend import MongoDBBackend, UnQLiteBackend

STATUS_WAITING = "waiting"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_PART_DONE = "part_done"
STATUS_FAILED = "failed"


DB_BACKEND_MAP = {
    "mongodb": MongoDBBackend,
    "unqlite": UnQLiteBackend,
}


class TaskManager:
    ENCODE_FIELDS_PREFIX = ["def", "res"]

    def __init__(self, config: GlobalConfig, task_pool: str):
        self.logger = get_logger(self)
        self.config = config
        db_backend, db_path = config.job.machine.taskdb_uri.split("://")[0:2]
        if db_backend == "mongodb":
            self.db_backend = MongoDBBackend(
                uri=self.config.job.machine.taskdb_uri,
                dbname=self.config.job.machine.taskdb_name,
            )
        elif db_backend == "unqlite":
            self.db_backend = UnQLiteBackend(database_path=db_path)
        self.task_pool = task_pool

    def _encode_task(self, task: Dict) -> Dict:
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    if isinstance(self.db_backend, MongoDBBackend):
                        task[k] = Binary(
                            pickle.dumps(
                                task[k], protocol=pickle.HIGHEST_PROTOCOL
                            )
                        )
                    else:
                        task[k] = base64.b64encode(
                            pickle.dumps(
                                task[k], protocol=pickle.HIGHEST_PROTOCOL
                            )
                        ).decode("ascii")

        return task

    def _decode_task(self, task: Dict) -> Dict:
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    if isinstance(self.db_backend, UnQLiteBackend):
                        task[k] = pickle.loads(
                            base64.b64decode(task[k].encode("ascii"))
                        )
                    else:
                        task[k] = pickle.loads(task[k])
        return task

    def insert_task(self, task: Dict) -> Any:
        task = self._encode_task(task)
        return self.db_backend.insert(self.task_pool, task)

    def query(self, query: Dict = None, decode: bool = True) -> Iterable[Dict]:
        if query is None:
            query = {}
        results = self.db_backend.find(self.task_pool, query)
        for res in results:
            if decode:
                yield self._decode_task(res)
            else:
                yield res

    def commit_task_res(
        self, task_id: str, res: Any, status: str = STATUS_DONE
    ) -> None:
        if isinstance(self.db_backend, MongoDBBackend):
            res_str = Binary(
                pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
            )
            task_obj_id = ObjectId(task_id)
        else:
            res_str = base64.b64encode(
                pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
            ).decode("ascii")
            task_obj_id = task_id
        update = {
            "$set": {
                "status": status,
                "res": res_str,
            }
        }
        query = {"_id": task_obj_id}
        self.db_backend.update_one(self.task_pool, query, update)

    def insert_task_def(
        self, task_def: tp.Union[omegaconf.DictConfig, Dict], **update_dict
    ) -> Any:
        task_def_container = (
            OmegaConf.to_container(task_def, resolve=True)
            if isinstance(task_def, omegaconf.DictConfig)
            else task_def
        )
        task_def_container.update(update_dict)
        task = {
            "def": task_def,
            "filter": task_def_container,
            "status": STATUS_WAITING,
        }
        encoded_task = self._encode_task(task)
        return self.db_backend.insert(self.task_pool, encoded_task)

    def insert_task(self, task: Dict) -> Any:
        encoded_task = self._encode_task(task)
        return self.db_backend.insert(self.task_pool, encoded_task)

    def create_task(
        self,
        task_def_l: List[ExperimentConfig],
        dry_run: bool = False,
        print_nt: bool = False,
        repeat_index: int = 1,
        total_repeat: int = 1,
    ) -> List[str]:
        new_tasks = []
        _id_list = []
        for task_def in task_def_l:
            task_dict = OmegaConf.to_container(task_def, resolve=True)
            filter_query = {"filter": task_dict}
            existing_task = self.db_backend.find_one(
                self.task_pool, filter_query
            )
            if existing_task is None:
                new_tasks.append(task_dict)
                if not dry_run:
                    insert_result = self.insert_task_def(
                        task_def,
                        repeat_index=repeat_index,
                        total_repeat=total_repeat,
                    )
                    if isinstance(self.db_backend, MongoDBBackend):
                        _id_list.append(str(insert_result.inserted_id))
                    else:
                        _id_list.append(insert_result)
                else:
                    _id_list.append(None)
            else:
                if print_nt:
                    print(
                        f"Task already exists, it is: {task_dict}\n"
                        f"In MongoDB it is: {self._decode_task(existing_task)}"
                    )
                _id_list.append(str(existing_task["_id"]))
        return [] if dry_run else _id_list

    def run_tasks(
        self,
        task_func: Callable,
        query: dict = None,
        num_workers: int = 1,
        before_status: str = STATUS_WAITING,
        after_status: str = STATUS_DONE,
        **kwargs,
    ):
        if query is None:
            query = {}
        executor = loky.get_reusable_executor(max_workers=num_workers)
        local_query = query.copy()
        local_query["status"] = before_status
        tasks = list(self.query(query=local_query, decode=True))

        jobs = []
        for task in tasks:
            param = (
                task["def"] if before_status == STATUS_WAITING else task["res"]
            )
            if isinstance(self.db_backend, UnQLiteBackend):
                task["_id"] = task["__id"]
            task_kwargs = {**kwargs, "task_mongodb_id": str(task["_id"])}
            self.db_backend.update_one(
                self.task_pool,
                {"_id": task["_id"]},
                {"$set": {"status": STATUS_RUNNING}},
            )
            if num_workers <= 1:
                res = task_func(param, **task_kwargs)
                self.commit_task_res(str(task["_id"]), res, status=after_status)
            else:
                self.logger.info(f"Submitting task {task['_id']} to executor")
                job = executor.submit(
                    task_func, param, **task_kwargs, is_subprocess=True
                )
                jobs.append((job, task))

        if num_workers > 1:
            self.wait_jobs(jobs)

    def wait_jobs(self, jobs: List[Tuple[loky.Future, Dict]]):
        while jobs:
            done_jobs = []
            for i, (job, task) in enumerate(jobs):
                if job.done():
                    # We do not use try-except to let the error expose itself
                    try:
                        res = job.result()
                        self.commit_task_res(str(task["_id"]), res)
                    except Exception as e:
                        self.logger.error(f"Error in task {task['_id']}: {e}")
                        self.commit_task_res(
                            str(task["_id"]), None, status=STATUS_FAILED
                        )
                        raise e
                    done_jobs.append(i)

            # Remove completed jobs from the list
            for idx in reversed(done_jobs):
                del jobs[idx]

            time.sleep(30)  # Sleep for a short duration before checking again
