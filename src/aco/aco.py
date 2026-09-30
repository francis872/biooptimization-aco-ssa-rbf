import numpy as np

def nearest_neighbor_tour(dist, start=0):
    n=len(dist); route=[start]; unseen=set(range(n))-{start}
    while unseen:
        j=min(unseen,key=lambda x:dist[route[-1],x]); route.append(j); unseen.remove(j)
    return route, float(sum(dist[route[i],route[(i+1)%n]] for i in range(n)))

class AntColonyTSP:
    def __init__(self, distances, n_ants=30, iterations=100, alpha=1, beta=2, rho=.5, seed=42):
        self.d=np.asarray(distances,float); self.n=len(self.d); self.n_ants=n_ants; self.iterations=iterations
        self.alpha=alpha; self.beta=beta; self.rho=rho; self.rng=np.random.default_rng(seed)
    def run(self):
        tau=np.ones_like(self.d); eta=1/(self.d+1e-12); np.fill_diagonal(eta,0)
        best=None; best_len=float("inf"); conv=[]
        for _ in range(self.iterations):
            tours=[]
            for _ in range(self.n_ants):
                start=int(self.rng.integers(self.n)); route=[start]; unseen=set(range(self.n))-{start}
                while unseen:
                    cand=np.array(list(unseen)); w=(tau[route[-1],cand]**self.alpha)*(eta[route[-1],cand]**self.beta)
                    p=w/w.sum() if w.sum()>0 else np.ones(len(cand))/len(cand)
                    j=int(self.rng.choice(cand,p=p)); route.append(j); unseen.remove(j)
                length=float(sum(self.d[route[i],route[(i+1)%self.n]] for i in range(self.n)))
                tours.append((route,length))
                if length<best_len: best,best_len=route.copy(),length
            tau*=1-self.rho
            for route,length in tours:
                for i in range(self.n):
                    a,b=route[i],route[(i+1)%self.n]; tau[a,b]+=1/length; tau[b,a]+=1/length
            conv.append(best_len)
        return {"route":best,"distance":best_len,"convergence":conv}
