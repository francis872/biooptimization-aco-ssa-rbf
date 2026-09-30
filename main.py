import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.aco.aco import AntColonyTSP, nearest_neighbor_tour
from src.ssa.ssa import SalpSwarm, rastrigin
from src.experiment import run_seed
from src.database.mongodb import MongoRunStore, build_run_document

OUT=Path("results"); OUT.mkdir(exist_ok=True)

def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8")

def exercise_aco():
    rng=np.random.default_rng(42); xy=rng.random((20,2))*100; d=np.sqrt(((xy[:,None]-xy[None,:])**2).sum(2))
    configs=[{"alpha":1,"beta":2,"rho":.5},{"alpha":2,"beta":5,"rho":.1}]; rows=[]
    for c in configs:
        t=time.perf_counter(); r=AntColonyTSP(d,30,100,**c,seed=42).run(); r.update(c); r["time_s"]=time.perf_counter()-t; rows.append(r)
    nn=nearest_neighbor_tour(d); result={"configs":rows,"nearest_neighbor":{"route":nn[0],"distance":nn[1]}}; save("aco_tsp_results.json",result); return result

def exercise_ssa():
    t=time.perf_counter(); r=SalpSwarm(rastrigin,10,-5.12,5.12,30,100,42).run(); r["time_s"]=time.perf_counter()-t; save("ssa_rastrigin_results.json",r); return r

def dataset():
    df=pd.read_csv("data/sdss_sample.csv"); target="redshift" if "redshift" in df.columns else df.select_dtypes("number").columns[-1]
    nums=[c for c in df.select_dtypes("number").columns if c!=target][:8]; X=df[nums].replace([np.inf,-np.inf],np.nan).dropna(); y=df.loc[X.index,target]
    X1,Xt,y1,yt=train_test_split(X,y,test_size=.2,random_state=42); Xtr,Xv,ytr,yv=train_test_split(X1,y1,test_size=.25,random_state=42)
    s=StandardScaler().fit(Xtr); return s.transform(Xtr),ytr.to_numpy(),s.transform(Xv),yv.to_numpy(),s.transform(Xt),yt.to_numpy()

def exercise_rbf():
    Xtr,ytr,Xv,yv,Xt,yt=dataset(); seeds=[11,22,33,44,55,66,77,88,99,110]; runs=[]; store=None
    try: store=MongoRunStore(); store.ping(); store.ensure_indexes(); print("MongoDB Atlas conectado")
    except Exception as e: print("MongoDB no disponible:",e)
    for seed in seeds:
        r=run_seed(Xtr,ytr,Xv,yv,Xt,yt,seed); runs.append(r); print(seed,r["metrics"]["rmse"])
        if store:
            doc=build_run_document(algorithm=r["algorithm"],problem="RBF regression optimized with SSA",dataset="SDSS",seed=seed,params=r["params"],metrics={"rmse":r["metrics"]["rmse"],"baseline_rmse":r["metrics"]["baseline_rmse"]},evaluations=r["evaluations"],convergence=r["convergence"],time_s=r["metrics"]["time_s"],experiment_id="rbf-ssa-10-seeds")
            store.insert_run(doc)
    vals=np.array([r["metrics"]["rmse"] for r in runs]); summary={"mean":float(vals.mean()),"std":float(vals.std()),"median":float(np.median(vals)),"min":float(vals.min()),"max":float(vals.max())}
    out={"ssa_rbf":{"per_seed":runs,"summary":summary}}; save("rbf_ssa_results.json",out); return out

if __name__=="__main__":
    print("Ejercicio 1: ACO/TSP"); exercise_aco()
    print("Ejercicio 2: SSA/Rastrigin"); exercise_ssa()
    print("Ejercicio 3: RBF + SSA, 10 semillas"); exercise_rbf()
    print("Resultados guardados en results/")
