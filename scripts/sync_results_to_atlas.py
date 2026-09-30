from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri=os.getenv("MONGO_URI")
if not uri:
    raise SystemExit("MONGO_URI no esta definida. Copia .env.example a .env y completa la URI.")
db_name=os.getenv("MONGO_DB","biooptimization")
collection_name=os.getenv("MONGO_COLLECTION","runs")
path=Path("results/rbf_ssa_results.json")
if not path.exists():
    raise SystemExit("No existe results/rbf_ssa_results.json. Ejecuta python main.py primero.")

data=json.loads(path.read_text(encoding="utf-8"))
client=MongoClient(uri,serverSelectionTimeoutMS=10000)
client.admin.command("ping")
col=client[db_name][collection_name]
col.create_index([("experiment_id",1),("algorithm",1),("seed",1)],unique=True,name="experiment_algorithm_seed")
col.create_index([("metrics.rmse",1)],name="rmse")

def doc(algorithm, r):
    metrics={
        "mse": float(r["test_mse"]),
        "rmse": float(r["test_rmse"]),
        "r2": float(r["test_r2"]),
        "time_s": float(r["time_s"]),
    }
    return {
        "algorithm": algorithm,
        "problem": "redshift_regression",
        "dataset": "sdss_sample.csv",
        "seed": int(r["seed"]),
        "params": r["params"],
        "metrics": metrics,
        "evaluations": int(r.get("evaluations",0)),
        "convergence": [float(x) for x in r.get("convergence",[])],
        "created_at": datetime.now(timezone.utc),
        "experiment_id": "rbf_vs_ssa_ex3",
    }

count=0
for algorithm,key in [("rbf_baseline","baseline_rbf"),("ssa_rbf","ssa_rbf")]:
    for r in data[key]["per_seed"]:
        d=doc(algorithm,r)
        col.replace_one(
            {"experiment_id":d["experiment_id"],"algorithm":algorithm,"seed":d["seed"]},
            d,
            upsert=True,
        )
        count+=1
print(f"OK: {count} documentos sincronizados en {db_name}.{collection_name}")
print("Total documentos experimento:",col.count_documents({"experiment_id":"rbf_vs_ssa_ex3"}))
