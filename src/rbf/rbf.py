import numpy as np
from sklearn.cluster import KMeans

class RBFRegressor:
    def __init__(self, n_centers=10, sigma=1.0, reg_lambda=.001, seed=42):
        self.n_centers=n_centers; self.sigma=sigma; self.reg_lambda=reg_lambda; self.seed=seed
    def _phi(self,X):
        d=((X[:,None,:]-self.centers[None,:,:])**2).sum(2)
        return np.exp(-d/(2*self.sigma**2))
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float)
        self.centers=KMeans(n_clusters=min(self.n_centers,len(X)),random_state=self.seed,n_init=5).fit(X).cluster_centers_
        P=self._phi(X); self.weights=np.linalg.solve(P.T@P+self.reg_lambda*np.eye(P.shape[1]),P.T@y); return self
    def predict(self,X): return self._phi(np.asarray(X,float))@self.weights
