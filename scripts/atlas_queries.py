from __future__ import annotations
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
client=MongoClient(os.environ["MONGO_URI"],serverSelectionTimeoutMS=10000)
client.admin.command("ping")
col=client[os.getenv("MONGO_DB","biooptimization")][os.getenv("MONGO_COLLECTION","runs")]
flt={"experiment_id":"rbf-ssa-10-seeds","algorithm":"ssa_rbf"}

print("\n1) MEJOR CORRIDA SSA-RBF")
best=col.find_one(flt,sort=[("metrics.rmse",1)])
if best:
    print({"seed":best["seed"],"rmse":best["metrics"]["rmse"],"params":best["params"]})
else:
    print("No se encontraron corridas.")

print("\n2) ERROR PROMEDIO DE LAS 10 CORRIDAS SSA-RBF")
rows=list(col.aggregate([
    {"$match":flt},
    {"$group":{"_id":"$algorithm","runs":{"$sum":1},"avg_rmse":{"$avg":"$metrics.rmse"},"std_rmse":{"$stdDevPop":"$metrics.rmse"}}}
]))
print(rows[0] if rows else "Sin datos")

print("\n3) RESUMEN DEL EXPERIMENTO")
for row in col.aggregate([
    {"$match":flt},
    {"$group":{"_id":"$algorithm","runs":{"$sum":1},"mean_rmse":{"$avg":"$metrics.rmse"},"std_rmse":{"$stdDevPop":"$metrics.rmse"},"min_rmse":{"$min":"$metrics.rmse"},"max_rmse":{"$max":"$metrics.rmse"},"avg_time_s":{"$avg":"$metrics.time_s"}}}
]):
    print(row)

print("\n4) TOP 5 CONFIGURACIONES POR RMSE")
for row in col.find(flt,{"_id":0,"seed":1,"params":1,"metrics.rmse":1,"metrics.baseline_rmse":1}).sort("metrics.rmse",1).limit(5):
    print(row)
