from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, Iterable

from dotenv import load_dotenv

load_dotenv()

REQUIRED_FIELDS = {
    "algorithm", "problem", "dataset", "seed", "params", "metrics",
    "evaluations", "convergence", "created_at"
}

def build_run_document(*, algorithm: str, problem: str, dataset: str, seed: int,
                       params: Dict[str, Any], metrics: Dict[str, Any], evaluations: int,
                       convergence: Iterable[float], time_s: float, experiment_id: str | None = None) -> Dict[str, Any]:
    doc = {
        "algorithm": algorithm,
        "problem": problem,
        "dataset": dataset,
        "seed": int(seed),
        "params": params,
        "metrics": {**metrics, "time_s": float(time_s)},
        "evaluations": int(evaluations),
        "convergence": [float(x) for x in convergence],
        "created_at": datetime.now(timezone.utc),
    }
    if experiment_id:
        doc["experiment_id"] = experiment_id
    return doc

def validate_run_document(doc: Dict[str, Any]) -> bool:
    missing = REQUIRED_FIELDS - set(doc)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    return True

class MongoRunStore:
    def __init__(self, uri: str | None = None, db_name: str | None = None, collection: str | None = None):
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGO_DB", "biooptimization")
        self.collection_name = collection or os.getenv("MONGO_COLLECTION", "runs")
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError("pymongo is required for MongoDB operations. Install requirements.txt") from exc
        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)

    @property
    def collection(self):
        return self.client[self.db_name][self.collection_name]

    def ping(self) -> bool:
        self.client.admin.command("ping")
        return True

    def safe_uri(self) -> str:
        if "@" not in self.uri:
            return self.uri
        prefix, suffix = self.uri.rsplit("@", 1)
        scheme = prefix.split("://", 1)[0]
        return f"{scheme}://***:***@{suffix}"

    def ensure_indexes(self) -> None:
        self.collection.create_index([("algorithm", 1), ("created_at", -1)], name="algorithm_created_at")
        self.collection.create_index([("experiment_id", 1), ("seed", 1), ("algorithm", 1)], name="experiment_seed_algorithm")
        self.collection.create_index([("metrics.rmse", 1)], name="rmse")

    def insert_run(self, doc: Dict[str, Any]):
        validate_run_document(doc)
        return self.collection.insert_one(doc).inserted_id

    def best_run(self, algorithm: str, metric: str = "rmse"):
        return self.collection.find_one({"algorithm": algorithm}, sort=[(f"metrics.{metric}", 1)])

    def aggregate_by_algorithm(self):
        pipeline = [
            {"$group": {
                "_id": "$algorithm",
                "runs": {"$sum": 1},
                "avg_rmse": {"$avg": "$metrics.rmse"},
                "min_rmse": {"$min": "$metrics.rmse"},
                "max_rmse": {"$max": "$metrics.rmse"},
                "avg_time_s": {"$avg": "$metrics.time_s"},
            }},
            {"$sort": {"_id": 1}},
        ]
        return list(self.collection.aggregate(pipeline))

    def log_genetic_run(
        self,
        algorithm: str,
        problem: str,
        dataset: str,
        seed: int,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        evaluations: int,
        convergence: Iterable[float],
        time_s: float,
        experiment_id: str | None = None,
    ) -> str:
        doc = build_run_document(
            algorithm=algorithm,
            problem=problem,
            dataset=dataset,
            seed=seed,
            params=params,
            metrics=metrics,
            evaluations=evaluations,
            convergence=convergence,
            time_s=time_s,
            experiment_id=experiment_id,
        )
        return str(self.insert_run(doc))

    def get_recent_runs(self, limit: int = 10):
        return list(self.collection.find().sort("created_at", -1).limit(limit))

