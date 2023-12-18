from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

import unqlite
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

STATUS_WAITING = "waiting"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_PART_DONE = "part_done"
STATUS_FAILED = "failed"


class DatabaseBackend(ABC):
    @abstractmethod
    def insert(self, collection: str, document: Dict) -> Any:
        pass

    @abstractmethod
    def find_one(self, collection: str, query: Dict) -> Dict:
        pass

    @abstractmethod
    def find(self, collection: str, query: Dict) -> Iterable[Dict]:
        pass

    @abstractmethod
    def update_one(self, collection: str, query: Dict, update: Dict) -> Any:
        pass

    @abstractmethod
    def replace_one(
        self, collection: str, query: Dict, replacement: Dict
    ) -> Any:
        pass


class UnQLiteBackend(DatabaseBackend):
    def __init__(self, database_path: str):
        self.db = unqlite.UnQLite(database_path)
        self.collections: Dict[
            str, unqlite.Collection
        ] = {}  # To manage different collections

    def _get_collection(self, collection: str):
        if collection not in self.collections:
            self.collections[collection] = self.db.collection(collection)
            if not self.collections[collection].exists():
                self.collections[collection].create()
        return self.collections[collection]

    def insert(self, collection: str, document: Dict) -> Any:
        col = self._get_collection(collection)
        return col.store(document)

    def _matches_query(self, doc: Dict, query: Dict) -> bool:
        for key, value in query.items():
            if key == "_id":
                key = "__id"
            if isinstance(value, dict) and "$in" in value:
                if doc.get(key) not in value["$in"]:
                    return False
            elif doc.get(key) != value:
                return False
        return True

    def find_one(self, collection: str, query: Dict) -> Dict:
        col = self._get_collection(collection)
        for doc in col:
            if self._matches_query(doc, query):
                return doc
        return None

    def find(self, collection: str, query: Dict) -> Iterable[Dict]:
        col = self._get_collection(collection)
        return [doc for doc in col if self._matches_query(doc, query)]

    def update_one(self, collection: str, query: Dict, update: Dict) -> Any:
        col = self._get_collection(collection)
        with self.db.transaction():
            for doc in col:
                if self._matches_query(doc, query):
                    for update_key, update_value in update.get(
                        "$set", {}
                    ).items():
                        doc[update_key] = update_value
                    col.update(doc["__id"], doc)
                    return doc
        return None

    def replace_one(
        self, collection: str, query: Dict, replacement: Dict
    ) -> Any:
        col = self._get_collection(collection)
        with self.db.transaction():
            for doc in col:
                if self._matches_query(doc, query):
                    for k in list(doc.keys()):
                        if k not in replacement:
                            del doc[k]
                    doc.update(replacement)
                    col.update(doc["__id"], doc)
                    return doc
        return None


class MongoDBBackend(DatabaseBackend):
    def __init__(self, uri: str, dbname: str):
        self.client = MongoClient(uri)
        self.db = self.client[dbname]

    def _get_collection(self, collection: str) -> Collection:
        return self.db[collection]

    def insert(self, collection: str, document: Dict) -> Any:
        col = self._get_collection(collection)
        return col.insert_one(document)

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

    def find_one(self, collection: str, query: Dict) -> Dict:
        col = self._get_collection(collection)
        return col.find_one(query)

    def find(self, collection: str, query: Dict) -> Iterable[Dict]:
        col = self._get_collection(collection)
        query = query.copy()
        query = self._decode_query(query)
        return col.find(query)

    def update_one(self, collection: str, query: Dict, update: Dict) -> Any:
        col = self._get_collection(collection)
        return col.update_one(query, update)

    def replace_one(
        self, collection: str, query: Dict, replacement: Dict
    ) -> Any:
        col = self._get_collection(collection)
        return col.replace_one(query, replacement)
