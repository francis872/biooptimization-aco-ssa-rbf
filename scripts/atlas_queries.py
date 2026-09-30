from __future__ import annotations
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
client=MongoClient(os.environ["MONGO_URI"],serverSelectionTimeoutMS=10000)
client.admin.command("ping")
col=client[os.getenv("MONGO_DB","biooptimization")][os.getenv("MONGO_COLLECTION","runs")]
flt={"experiment_id":"rbf_vs_ssa_ex3"}

print("\n1) MEJOR CORRIDA SSA-RBF")
best=col.find_one({**flt,"algorithm":"ssa_rbf"},sort=[("metrics.rmse",1)])
print({"seed":best["seed"],"rmse":best["metrics"]["rmse"],"params":best["params"]})

print("\n2) ERROR PROMEDIO DE LAS 10 CORRIDAS SSA-RBF")
avg=list(col.aggregate([
    {"$match":{**flt,"algorithm":"ssa_rbf"}},
    {"$group":{"_id":"$algorithm","runs":{"$sum":1},"avg_rmse":{"$avg":"$metrics.rmse"}}},
]))[0]
print(avg)

print("\n3) AGRUPACION POR ALGORITMO")
for row in col.aggregate([
    {"$match":flt},
    {"$group":{
        "_id":"$algorithm",
        "runs":{"$sum":1},
        "mean_rmse":{"$avg":"$metrics.rmse"},
        "min_rmse":{"$min":"$metrics.rmse"},
        "max_rmse":{"$max":"$metrics.rmse"},
        "avg_time_s":{"$avg":"$metrics.time_s"},
    }},
    {"$sort":{"_id":1}},
]):
    print(row)

print("\n4) CONFIGURACION SSA-RBF CON BUEN ERROR Y VARIABILIDAD")
for row in col.find({**flt,"algorithm":"ssa_rbf"},{"_id":0,"seed":1,"params":1,"metrics.rmse":1}).sort("metrics.rmse",1).limit(5):
    print(row)
