import time, numpy as np
from sklearn.metrics import mean_squared_error
from src.rbf.rbf import RBFRegressor
from src.ssa.ssa import SalpSwarm

def optimize_rbf(Xtr,ytr,Xv,yv,seed=42,iterations=25,population=15):
    def objective(z):
        M=int(np.clip(round(z[0]),3,30)); sigma=float(np.clip(z[1],.01,5)); lam=float(10**np.clip(z[2],-6,0))
        m=RBFRegressor(M,sigma,lam,seed).fit(Xtr,ytr)
        return mean_squared_error(yv,m.predict(Xv))
    s=SalpSwarm(objective,3,[3,.01,-6],[30,5,0],population,iterations,seed).run()
    z=s["best_position"]; return {"M":int(round(z[0])),"sigma":float(z[1]),"lambda":float(10**z[2]),"best_fitness":s["best_fitness"],"convergence":s["convergence"],"evaluations":s["evaluations"]}

def run_seed(Xtr,ytr,Xv,yv,Xt,yt,seed=42):
    t=time.perf_counter(); base=RBFRegressor(10,1,.001,seed).fit(Xtr,ytr); base_rmse=float(mean_squared_error(yt,base.predict(Xt))**.5)
    opt=optimize_rbf(Xtr,ytr,Xv,yv,seed); model=RBFRegressor(opt["M"],opt["sigma"],opt["lambda"],seed).fit(Xtr,ytr)
    rmse=float(mean_squared_error(yt,model.predict(Xt))**.5)
    return {"algorithm":"ssa_rbf","seed":seed,"params":{k:opt[k] for k in ("M","sigma","lambda")},"metrics":{"rmse":rmse,"baseline_rmse":base_rmse,"time_s":time.perf_counter()-t},"evaluations":opt["evaluations"],"convergence":opt["convergence"]}
