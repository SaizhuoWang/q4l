"""Adapted version of trainer and task manager from qlib.

Provides task management capability with a MongoDB backend. The MongoDB backend
serves as a database for tracking the status of tasks. The task manager is
responsible for managing the tasks and their status.

"""

import pickle
import time
import traceback
import typing
import typing as tp
from contextlib import contextmanager
from typing import Callable, List

import loky
import omegaconf
import pymongo
from bson.binary import Binary
from bson.objectid import ObjectId
from omegaconf import OmegaConf
from pymongo.errors import InvalidDocument

# from qlib.workflow.task.manage import TaskManager
from tqdm import tqdm

from q4l.utils.log import get_logger

from ..config import ExperimentConfig
from ..qlib.config import C
from ..qlib.workflow import R
from ..qlib.workflow.task.utils import get_mongodb

STATUS_WAITING = "waiting"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_PART_DONE = "part_done"


class TaskManager:
    """MongoDB Task Manager for updating and querying tasks. Task pool is the
    collection in a MongoDB that stores tasks. Each task is a document in the
    collection. Document fields explained:

    - `_id`: Object ID of the task that uniquely identifies the task.
    - `def`: Pickle-serialized bytes of the task definition. This can be unpickled into a
        `omegaconf.DictConfig` object.
    - `filter`: A dictionary storing task-related configs in plain-text. Filters
        are attributes of a task document that can be used for further querying use
        (as conditions in filters). In addition to the task definition, the filter dict also
        contains some index information (e.g. repeat index, rolling index) that are used for querying.
        Such information is appended at task runtime, not defined in the task definition (YAML/cli).
    - `status`: A string that indicates the status of the task. It can be one of the status
        that are listed below in the class definition.
    - `res`: Pickle-serialized bytes of the task result. This can be deserialized into a dictionary
        that stores the task result defined by task-specific routines.
    - `checkpoint`: Pickle-serialized bytes of the task checkpoint. This can be deserialized into
        a dictionary that can be further used for resuming the task following the defined protocol
        in task-specific routines. Note that this checkpoint do not store big data such as model
        parameters. Instead it store the path/pointer to these files. Same information can also
        be found in mlflow, and duplication here is for robustness.

    """

    STATUS_WAITING = "waiting"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_PART_DONE = "part_done"
    STATUS_FAILED = "failed"

    # Those tag will help you distinguish whether the Recorder has finished traning
    STATUS_KEY = "train_status"
    STATUS_BEGIN = "begin_task_train"
    STATUS_END = "end_task_train"

    # This tag is the _id in TaskManager to distinguish tasks.
    TM_ID = "_id in TaskManager"

    # These fields will be encoded and decoded by pickle.
    ENCODE_FIELDS_PREFIX = ["def", "res"]

    def __init__(self, task_pool: str):
        """Init Task Manager, remember to make the statement of MongoDB url and
        database name firstly. A TaskManager instance serves a specific task
        pool. The static method of this module serves the whole MongoDB.

        Parameters
        ----------
        task_pool: str
            the name of Collection in MongoDB

        """
        self.task_pool: pymongo.collection.Collection = getattr(
            get_mongodb(), task_pool
        )
        self.logger = get_logger(self)
        self.logger.info(f"Collection {task_pool} is used to store tasks.")

    @staticmethod
    def list() -> list:
        """List the all collection(task_pool) of the db.

        Returns:
            list

        """
        return get_mongodb().list_collection_names()

    def _encode_task(self, task):
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = Binary(
                        pickle.dumps(task[k], protocol=C.dump_protocol_version)
                    )
        return task

    def _decode_task(self, task):
        """_decode_task is Serialization tool. Mongodb needs JSON, so it needs
        to convert Python objects into JSON objects through pickle.

        Parameters
        ----------
        task : dict
            task information

        Returns
        -------
        dict
            JSON required by mongodb

        """
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = pickle.loads(task[k])
        return task

    def _dict_to_str(self, flt):
        return {k: str(v) for k, v in flt.items()}

    def _decode_query(self, query):
        """If the query includes any `_id`, then it needs `ObjectId` to decode.
        For example, when using TrainerRM, it needs query `{"_id": {"$in":
        _id_list}}`. Then we need to `ObjectId` every `_id` in `_id_list`.

        Args:
            query (dict): query dict. Defaults to {}.

        Returns:
            dict: the query after decoding.

        """
        if "_id" in query:
            if isinstance(query["_id"], dict):
                for key in query["_id"]:
                    query["_id"][key] = [ObjectId(i) for i in query["_id"][key]]
            else:
                query["_id"] = ObjectId(query["_id"])
        return query

    def replace_task(self, task, new_task):
        """Use a new task to replace a old one.

        Args:
            task: old task
            new_task: new task

        """
        new_task = self._encode_task(new_task)
        query = {"_id": ObjectId(task["_id"])}
        try:
            self.task_pool.replace_one(query, new_task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            self.task_pool.replace_one(query, new_task)

    def insert_task(self, task):
        """Insert a task.

        Args:
            task: the task waiting for insert

        Returns:
            pymongo.results.InsertOneResult

        """
        try:
            insert_result = self.task_pool.insert_one(task)
        except InvalidDocument:
            task["filter"] = self._dict_to_str(task["filter"])
            insert_result = self.task_pool.insert_one(task)
        return insert_result

    def insert_task_def(
        self,
        task_def: typing.Union[omegaconf.DictConfig, typing.Dict],
        **update_dict,
    ):
        """Insert a task to task_pool.

        Parameters
        ----------
        task_def: dict
            the task definition

        Returns
        -------
        pymongo.results.InsertOneResult

        """
        task_def_container: tp.Dict = (
            OmegaConf.to_container(task_def, resolve=True)
            if isinstance(task_def, omegaconf.DictConfig)
            else task_def
        )
        # Add some additional information to the task filter
        task_def_container.update(update_dict)
        task = self._encode_task(
            {
                "def": task_def,
                "filter": task_def_container,  # FIXME: catch the raised error
                "status": self.STATUS_WAITING,
            }
        )
        insert_result = self.insert_task(task)
        return insert_result

    def create_task(
        self,
        task_def_l: List[ExperimentConfig],
        dry_run=False,
        print_nt=False,
        logger=None,
        repeat_index: int = 1,
        total_repeat: int = 1,
    ) -> List[str]:
        """If the tasks in task_def_l are new, then insert new tasks into the
        task_pool, and record inserted_id. If a task is not new, then just query
        its _id.

        Parameters
        ----------
        task_def_l: list
            a list of task
        dry_run: bool
            if insert those new tasks to task pool
        print_nt: bool
            if print new task

        Returns
        -------
        List[str]
            a list of the _id of task_def_l

        """
        self.repeat_index = repeat_index
        new_tasks = []
        _id_list = []
        for task in task_def_l:
            task_dict = OmegaConf.to_container(task, resolve=True)
            try:
                r = self.task_pool.find_one({"filter": task_dict})
            except InvalidDocument:
                r = self.task_pool.find_one(
                    {"filter": self._dict_to_str(task_dict)}
                )
            # When r is none, it indicates that r s a new task
            if r is None:
                new_tasks.append(task_dict)
                if not dry_run:
                    insert_result = self.insert_task_def(
                        task,
                        repeat_index=repeat_index,
                        total_repeat=total_repeat,
                    )
                    _id_list.append(insert_result.inserted_id)
                else:
                    _id_list.append(None)
            else:
                print(
                    f"task already exists, it is: {task_dict}\n"
                    f"In MongoDB it is : {self._decode_task(r)}"
                )
                _id_list.append(self._decode_task(r)["_id"])

        self.logger.info(
            f"Total Tasks: {len(task_def_l)}, New Tasks: {len(new_tasks)}"
        )

        if print_nt:  # print new task
            for task_dict in new_tasks:
                print(task_dict)

        return [] if dry_run else _id_list

    def fetch_task(self, query=None, status=STATUS_WAITING) -> dict:
        """Use query to fetch tasks.

        Args:
            query (dict, optional): query dict. Defaults to {}.
            status (str, optional): [description]. Defaults to STATUS_WAITING.

        Returns:
            dict: a task(document in collection) after decoding

        """
        if query is None:
            query = {}
        query = query.copy()
        query = self._decode_query(query)
        query.update({"status": status})
        task = self.task_pool.find_one_and_update(
            query,
            {"$set": {"status": self.STATUS_RUNNING}},
            sort=[("priority", pymongo.DESCENDING)],
        )
        # null will be at the top after sorting when using ASCENDING,
        # so the larger the number higher, the higher the priority
        if task is None:
            return None
        task["status"] = self.STATUS_RUNNING
        return self._decode_task(task)

    @contextmanager
    def safe_fetch_task(self, query=None, status=STATUS_WAITING):
        """Fetch task from task_pool using query with contextmanager.

        Parameters
        ----------
        query: dict
            the dict of query

        Returns
        -------
        dict: a task(document in collection) after decoding

        """
        if query is None:
            query = {}
        task = self.fetch_task(query=query, status=status)
        try:
            yield task
        except (
            Exception,
            KeyboardInterrupt,
        ):  # KeyboardInterrupt is not a subclass of Exception
            if task is not None:
                self.logger.info("Returning task before raising error")
                self.logger.info(f"The exception is:\n{traceback.format_exc()}")
                self.return_task(
                    task, status=status
                )  # return task as the original status
                self.logger.info("Task returned")
            raise

    def task_fetcher_iter(self, query=None):
        if query is None:
            query = {}
        while True:
            with self.safe_fetch_task(query=query) as task:
                if task is None:
                    break
                yield task

    def query(self, query=None, decode=True):
        """Query task in collection. This function may raise exception
        `pymongo.errors.CursorNotFound: cursor id not found` if it takes too
        long to iterate the generator.

        Parameters
        ----------
        query: dict
            the dict of query
        decode: bool

        Returns
        -------
        dict: a task(document in collection) after decoding

        """
        if query is None:
            query = {}
        query = query.copy()
        query = self._decode_query(query)
        for t in self.task_pool.find(query):
            yield self._decode_task(t)

    def re_query(self, _id) -> dict:
        """Use _id to query task.

        Args:
            _id (str): _id of a document

        Returns:
            dict: a task(document in collection) after decoding

        """
        t = self.task_pool.find_one({"_id": ObjectId(_id)})
        return self._decode_task(t)

    def commit_task_res(self, task, res, status=STATUS_DONE):
        """Commit the result to task['res'].

        Args:
            task ([type]): [description]
            res (object): the result you want to save
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE.
            Defaults to STATUS_DONE.

        """
        # A workaround to use the class attribute.
        if status is None:
            status = STATUS_DONE
        self.task_pool.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "status": status,
                    "res": Binary(
                        pickle.dumps(res, protocol=C.dump_protocol_version)
                    ),
                }
            },
        )

    def return_task(self, task, status=STATUS_WAITING):
        """Return a task to status. Always using in error handling.

        Args:
            task ([type]): [description]
            status (str, optional): STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_PART_DONE.
            Defaults to STATUS_WAITING.

        """
        if status is None:
            status = STATUS_WAITING
        update_dict = {"$set": {"status": status}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def remove(self, query=None):
        """Remove the task using query.

        Parameters
        ----------
        query: dict
            the dict of query

        """
        if query is None:
            query = {}
        query = query.copy()
        query = self._decode_query(query)
        self.task_pool.delete_many(query)

    def task_stat(self, query=None) -> dict:
        """Count the tasks in every status.

        Args:
            query (dict, optional): the query dict. Defaults to {}.

        Returns:
            dict

        """
        if query is None:
            query = {}
        query = query.copy()
        query = self._decode_query(query)
        tasks = self.query(query=query, decode=False)
        status_stat = {}
        for t in tasks:
            status_stat[t["status"]] = status_stat.get(t["status"], 0) + 1
        return status_stat

    def reset_waiting(self, query=None):
        """Reset all running task into waiting status. Can be used when some
        running task exit unexpected.

        Args:
            query (dict, optional): the query dict. Defaults to {}.

        """
        if query is None:
            query = {}
        query = query.copy()
        # default query
        if "status" not in query:
            query["status"] = self.STATUS_RUNNING
        return self.reset_status(query=query, status=self.STATUS_WAITING)

    def reset_status(self, query, status):
        query = query.copy()
        query = self._decode_query(query)
        print(self.task_pool.update_many(query, {"$set": {"status": status}}))

    def prioritize(self, task, priority: int):
        """Set priority for task.

        Parameters
        ----------
        task : dict
            The task query from the database
        priority : int
            the target priority

        """
        update_dict = {"$set": {"priority": priority}}
        self.task_pool.update_one({"_id": task["_id"]}, update_dict)

    def _get_undone_n(self, task_stat):
        return (
            task_stat.get(self.STATUS_WAITING, 0)
            + task_stat.get(self.STATUS_RUNNING, 0)
            + task_stat.get(self.STATUS_PART_DONE, 0)
        )

    def _get_total(self, task_stat):
        return sum(task_stat.values())

    def wait(self, query=None):
        """When multiprocessing, the main progress may fetch nothing from
        TaskManager because there are still some running tasks. So main progress
        should wait until all tasks are trained well by other progress or
        machines.

        Args:
            query (dict, optional): the query dict. Defaults to {}.

        """
        if query is None:
            query = {}
        task_stat = self.task_stat(query)
        total = self._get_total(task_stat)
        last_undone_n = self._get_undone_n(task_stat)
        if last_undone_n == 0:
            return
        self.logger.warning(
            f"Waiting for {last_undone_n} undone tasks. Please make sure they are running."
        )
        with tqdm(total=total, initial=total - last_undone_n) as pbar:
            while True:
                time.sleep(10)
                undone_n = self._get_undone_n(self.task_stat(query))
                pbar.update(last_undone_n - undone_n)
                last_undone_n = undone_n
                if undone_n == 0:
                    break

    def __str__(self):
        return f"TaskManager({self.task_pool})"

    def run_tasks(
        self,
        task_func: Callable,
        query: dict = None,
        num_workers: int = 1,
        before_status: str = STATUS_WAITING,
        after_status: str = STATUS_DONE,
        recorder_wrapper=R,
        **kwargs,
    ):
        """While the task pool is not empty (has WAITING tasks), use task_func
        to fetch and run tasks in task_pool.

        After running this method, here are 4 situations
        (before_status -> after_status):

            STATUS_WAITING -> STATUS_DONE: use task["def"] as
            `task_func` param, it means that the task has not been started

            STATUS_WAITING -> STATUS_PART_DONE: use task["def"] as
            `task_func` param

            STATUS_PART_DONE -> STATUS_PART_DONE: use task["res"] as
            `task_func` param, it means that the task has been started
            but not completed

            STATUS_PART_DONE -> STATUS_DONE: use task["res"] as
            `task_func` param

        """
        if query is None:
            query = {}
        executor = loky.get_reusable_executor(max_workers=num_workers)
        local_query = query.copy()
        local_query["status"] = before_status
        tasks = self.query(query=local_query, decode=True)

        jobs = []
        task_list = []
        for i, task in enumerate(tasks):
            task_list.append(task)
            # when fetching `WAITING` task, use task["def"] to train
            if before_status in [
                STATUS_WAITING,
                STATUS_RUNNING,
            ]:
                # param = task["def"]
                param = task["def"]
            elif before_status == STATUS_PART_DONE:
                param = task["res"]
            else:
                raise ValueError(
                    "The fetched task must be `STATUS_WAITING` "
                    "or `STATUS_PART_DONE`!"
                )

            kwargs["task_mongodb_id"] = task["_id"]
            kwargs["task_index"] = i
            # kwargs["task_manager"] = self
            kwargs["task_id_list"] = local_query["_id"][
                "$in"
            ]  # The task id list for in-func query

            self.task_pool.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": STATUS_RUNNING}},
            )
            if num_workers <= 1:
                res = task_func(param, **kwargs)
                self.commit_task_res(task=task, res=res)
            else:
                self.logger.info(f"Submitting task {task['_id']} to executor")
                job = executor.submit(
                    task_func, param, is_subprocess=True, **kwargs
                )
                jobs.append(job)

        # Wait for all workers to finish
        start_time = time.time()
        self.wait_jobs(jobs, task_list)
        # Fetch task results
        recs = []
        for _id in query["_id"]["$in"]:
            rec = self.re_query(_id)["res"]
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_BEGIN})
            rec.set_tags(**{self.TM_ID: _id})
            recs.append(rec)
        end_time = time.time()
        self.logger.info(f"Task running time: {end_time - start_time}")

    def wait_jobs(self, jobs: tp.List[loky.Future], task_list: tp.List[dict]):
        while jobs:
            done_jobs = 0
            for i, job in enumerate(jobs):
                if job.done():
                    res = job.result()
                    self.commit_task_res(task_list[i], res)
                    done_jobs += 1
            if done_jobs == len(jobs):
                break

            all_tasks = list(self.task_pool.find())
            all_rolling_tasks = [
                t
                for t in all_tasks
                if t["filter"]["repeat_index"] == self.repeat_index
            ]
            num_finished_tasks = len(
                [1 for t in all_rolling_tasks if t["status"] == "done"]
            )
            num_waiting_tasks = len(
                [1 for t in all_rolling_tasks if t["status"] == "waiting"]
            )
            num_running_tasks = len(
                [1 for t in all_rolling_tasks if t["status"] == "running"]
            )
            self.logger.info(
                f"Repeat {self.repeat_index}: "
                f"TOTAL {len(jobs)} | "
                f"FINISHED: {num_finished_tasks} "
                f"| RUNNING {num_running_tasks} | "
                f"WAITING {num_waiting_tasks}"
            )
            time.sleep(30)  # Sleep for a short duration before checking again
