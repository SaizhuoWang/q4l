import pickle
from abc import ABC, abstractmethod
from typing import Dict, Optional

from bson.binary import Binary
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import InvalidDocument


class DatabaseBackend(ABC):
    @abstractmethod
    def insert_task(self, task_id: str, task: Dict) -> None:
        pass

    @abstractmethod
    def fetch_task(self, task_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def update_task(self, task_id: str, updated_task: Dict) -> None:
        pass

    @abstractmethod
    def delete_task(self, task_id: str) -> None:
        pass

    @abstractmethod
    def list_tasks(self) -> Dict[str, Dict]:
        pass


class MongoDBDatabaseBackend(DatabaseBackend):
    ENCODE_FIELDS_PREFIX = ["def", "res"]

    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]

    def _encode_task(self, task: Dict) -> Dict:
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = Binary(pickle.dumps(task[k]))
        return task

    def _decode_task(self, task: Dict) -> Dict:
        for prefix in self.ENCODE_FIELDS_PREFIX:
            for k in list(task.keys()):
                if k.startswith(prefix):
                    task[k] = pickle.loads(task[k])
        return task

    def insert_task(self, task_id: str, task: Dict) -> None:
        task["_id"] = ObjectId(task_id)
        task = self._encode_task(task)
        try:
            self.collection.insert_one(task)
        except InvalidDocument:
            task["filter"] = {k: str(v) for k, v in task["filter"].items()}
            self.collection.insert_one(task)

    def fetch_task(self, task_id: str) -> Optional[Dict]:
        task = self.collection.find_one({"_id": ObjectId(task_id)})
        return self._decode_task(task) if task else None

    def update_task(self, task_id: str, updated_task: Dict) -> None:
        updated_task = self._encode_task(updated_task)
        self.collection.replace_one({"_id": ObjectId(task_id)}, updated_task)

    def delete_task(self, task_id: str) -> None:
        self.collection.delete_one({"_id": ObjectId(task_id)})

    def list_tasks(self) -> Dict[str, Dict]:
        tasks = self.collection.find()
        return {str(task["_id"]): self._decode_task(task) for task in tasks}


import pickle

from unqlite import UnQLite


class UnQLiteDatabaseBackend(DatabaseBackend):
    def __init__(self, db_path: str, task_pool: str):
        self.db = UnQLite(db_path)
        self.task_pool = task_pool

    def _encode_task(self, task: Dict) -> bytes:
        return pickle.dumps(task)

    def _decode_task(self, task_data: bytes) -> Dict:
        return pickle.loads(task_data)

    def _task_key(self, task_id: str) -> str:
        return f"{self.task_pool}:{task_id}"

    def insert_task(self, task_id: str, task: Dict) -> None:
        self.db[self._task_key(task_id)] = self._encode_task(task)

    def fetch_task(self, task_id: str) -> Optional[Dict]:
        task_data = self.db.get(self._task_key(task_id))
        return self._decode_task(task_data) if task_data else None

    def update_task(self, task_id: str, updated_task: Dict) -> None:
        self.insert_task(task_id, updated_task)

    def delete_task(self, task_id: str) -> None:
        del self.db[self._task_key(task_id)]

    def list_tasks(self) -> Dict[str, Dict]:
        prefix = f"{self.task_pool}:"
        return {
            key[len(prefix) :]: self._decode_task(value)
            for key, value in self.db
            if key.startswith(prefix)
        }


import pickle
from contextlib import contextmanager
from typing import Dict, List, Optional

from unqlite import UnQLite


class TaskManagerUnqLite:
    """UnQLite-based Task Manager for updating and querying tasks. Each task is
    stored as a key-value pair in the UnQLite database.

    The key is a combination of the task pool name and a unique task identifier.
    The value is a pickled dictionary containing the task information.

    Methods are provided for task manipulation including insertion, fetching,
    updating, deletion, and querying by various criteria.

    """

    STATUS_WAITING = "waiting"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_PART_DONE = "part_done"
    STATUS_FAILED = "failed"

    def __init__(self, db_path: str, task_pool: str):
        """Initialize the TaskManager with a path to the UnQLite database file
        and a task pool name.

        Parameters
        ----------
        db_path : str
            Path to the UnQLite database file.
        task_pool : str
            Name of the task pool, used as a namespace prefix for task keys.

        """
        self.db = UnQLite(db_path)
        self.task_pool = task_pool

    def _encode_task(self, task: Dict) -> bytes:
        """Encode a task dictionary into a pickled binary format."""
        return pickle.dumps(task)

    def _decode_task(self, task_data: bytes) -> Dict:
        """Decode a pickled binary format into a task dictionary."""
        return pickle.loads(task_data)

    def _task_key(self, task_id: str) -> str:
        """Generate a database key for a task using the task pool and task
        ID."""
        return f"{self.task_pool}:{task_id}"

    def insert_task(self, task_id: str, task: Dict) -> None:
        """Insert a task into the database.

        Parameters
        ----------
        task_id : str
            Unique identifier for the task.
        task : Dict
            Dictionary containing task details.

        """
        encoded_task = self._encode_task(task)
        self.db[self._task_key(task_id)] = encoded_task

    def fetch_task(self, task_id: str) -> Optional[Dict]:
        """Fetch a task from the database using its ID.

        Parameters
        ----------
        task_id : str
            Unique identifier for the task.

        Returns
        -------
        Optional[Dict]
            The task dictionary if found, None otherwise.

        """
        task_data = self.db.get(self._task_key(task_id))
        if task_data:
            return self._decode_task(task_data)
        return None

    def update_task(self, task_id: str, updated_task: Dict) -> None:
        """Update an existing task in the database.

        Parameters
        ----------
        task_id : str
            Unique identifier for the task.
        updated_task : Dict
            Updated dictionary containing task details.

        """
        self.insert_task(task_id, updated_task)

    def delete_task(self, task_id: str) -> None:
        """Delete a task from the database using its ID.

        Parameters
        ----------
        task_id : str
            Unique identifier for the task.

        """
        del self.db[self._task_key(task_id)]

    def list_tasks(self) -> Dict[str, Dict]:
        """List all tasks in the task pool.

        Returns
        -------
        Dict[str, Dict]
            A dictionary of task IDs and their corresponding task details.

        """
        prefix = f"{self.task_pool}:"
        return {
            key[len(prefix) :]: self._decode_task(value)
            for key, value in self.db
            if key.startswith(prefix)
        }

    @contextmanager
    def safe_fetch_task(self, task_id: str):
        """Context manager for safely fetching and updating a task.

        Parameters
        ----------
        task_id : str
            Unique identifier for the task.

        Yields
        ------
        Optional[Dict]
            The task dictionary if found, None otherwise.

        """
        task = self.fetch_task(task_id)
        try:
            yield task
        finally:
            if task:
                self.update_task(task_id, task)

    # Additional methods for querying, updating task status, etc., would be here.
    # They would involve iterating over all tasks in the pool and applying the
    # necessary logic in Python since UnQLite doesn't support complex querying.


class TaskManager:
    # ... [other methods and attributes]

    def insert_task_def(self, task_def: Dict, **update_dict) -> str:
        """Insert a task definition into the task pool.

        Parameters
        ----------
        task_def : Dict
            The task definition to insert.
        **update_dict
            Additional data to update the task definition with.

        Returns
        -------
        str
            The task ID of the inserted task.

        """
        task_def.update(update_dict)
        task = {
            "def": task_def,
            "filter": task_def,  # Adjust as per your filtering needs
            "status": self.STATUS_WAITING,
        }

        task_id = self._generate_unique_task_id()
        self.db.insert_task(task_id, task)
        return task_id

    def create_task(self, task_defs: List[Dict], dry_run=False) -> List[str]:
        """Create multiple tasks based on the provided task definitions.

        Parameters
        ----------
        task_defs : List[Dict]
            A list of task definitions to create tasks for.
        dry_run : bool
            If True, the tasks will not be actually inserted into the database.

        Returns
        -------
        List[str]
            A list of task IDs for the created tasks.

        """
        task_ids = []
        for task_def in task_defs:
            if not dry_run:
                task_id = self.insert_task_def(task_def)
            else:
                task_id = None  # Or generate a dummy ID for dry run scenarios
            task_ids.append(task_id)
        return task_ids

    def _generate_unique_task_id(self) -> str:
        """Generate a unique task ID.

        Returns
        -------
        str
            A unique task ID.

        """
        # This is a simple implementation. You might want to use a more robust method
        # to generate unique IDs, depending on your application's scale and requirements.
        import uuid

        return str(uuid.uuid4())
