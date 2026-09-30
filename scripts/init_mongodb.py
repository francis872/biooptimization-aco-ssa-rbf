from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.database.mongodb import MongoRunStore, build_run_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicializa MongoDB para biooptimization")
    parser.add_argument("--seed-sample", action="store_true", help="Inserta una corrida de prueba")
    args = parser.parse_args()

    load_dotenv()
    store = MongoRunStore()
    store.ping()
    store.ensure_indexes()

    result = {
        "status": "ok",
        "database": store.db_name,
        "collection": store.collection_name,
        "uri": store.safe_uri(),
        "indexes": store.collection.index_information(),
    }

    if args.seed_sample:
        doc = build_run_document(
            algorithm="deployment_check",
            problem="connectivity",
            dataset="none",
            seed=0,
            params={"purpose": "mongodb deployment verification"},
            metrics={"ok": 1.0},
            evaluations=1,
            convergence=[1.0],
            time_s=0.0,
            experiment_id="mongodb_deployment_check",
        )
        doc["deployment"] = {"initialized_at": datetime.now(timezone.utc)}
        result["sample_id"] = str(store.insert_run(doc))

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
