from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri=os.getenv("MONGO_URI")
if not uri:
    raise SystemExit("MONGO_URI no esta definida.")
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

runs=data.get("ssa_rbf",{}).get("per_seed",[])
if not runs:
    raise SystemExit("El JSON no contiene ssa_rbf.per_seed.")

count=0
for r in runs:
    metrics=r.get("metrics",{})
    d={
        "algorithm":"ssa_rbf",
        "problem":"redshift_regression",
        "dataset":"sdss_sample.csv",
        "seed":int(r["seed"]),
        "params":r.get("params",{}),
        "metrics":{
            "rmse":float(metrics["rmse"]),
            "baseline_rmse":float(metrics["baseline_rmse"]),
            "time_s":float(metrics.get("time_s",0)),
        },
        "evaluations":int(r.get("evaluations",0)),
        "convergence":[float(x) for x in r.get("convergence",[])],
        "created_at":datetime.now(timezone.utc),
        "experiment_id":"rbf-ssa-10-seeds",
    }
    col.replace_one(
        {"experiment_id":d["experiment_id"],"algorithm":d["algorithm"],"seed":d["seed"]},
        d,upsert=True
    )
    count+=1

print(f"OK: {count} documentos sincronizados en {db_name}.{collection_name}")
print("Total documentos experimento:",col.count_documents({"experiment_id":"rbf-ssa-10-seeds"}))
