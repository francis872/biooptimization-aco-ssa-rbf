import numpy as np

def rastrigin(x):
    x=np.asarray(x); return float(10*len(x)+np.sum(x*x-10*np.cos(2*np.pi*x)))

class SalpSwarm:
    def __init__(self, objective, dim, lb, ub, population=30, iterations=100, seed=42):
        self.f=objective; self.dim=dim; self.lb=np.broadcast_to(lb,(dim,)).astype(float); self.ub=np.broadcast_to(ub,(dim,)).astype(float)
        self.population=population; self.iterations=iterations; self.rng=np.random.default_rng(seed)
    def run(self):
        x=self.rng.uniform(self.lb,self.ub,(self.population,self.dim)); best=None; bf=float("inf"); conv=[]; evals=0
        for t in range(self.iterations):
            vals=np.array([self.f(v) for v in x]); evals+=len(vals); i=int(vals.argmin())
            if vals[i]<bf: bf=float(vals[i]); best=x[i].copy()
            c1=2*np.exp(-(4*(t+1)/self.iterations)**2)
            old=x.copy()
            for i in range(self.population):
                if i==0:
                    c2=self.rng.random(self.dim); c3=self.rng.random(self.dim)
                    x[i]=best+np.where(c3<.5,1,-1)*c1*((self.ub-self.lb)*c2+self.lb)
                else: x[i]=(old[i]+x[i-1])/2
            x=np.clip(x,self.lb,self.ub); conv.append(bf)
        return {"best_position":best,"best_fitness":bf,"convergence":conv,"evaluations":evals}
