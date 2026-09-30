import numpy as np
from src.aco.aco import AntColonyTSP
from src.ssa.ssa import SalpSwarm, rastrigin
from src.rbf.rbf import RBFRegressor

def test_aco_route_valid():
    rng=np.random.default_rng(1); p=rng.random((15,2)); d=np.sqrt(((p[:,None]-p[None,:])**2).sum(2))
    r=AntColonyTSP(d,n_ants=5,iterations=3,seed=1).run()["route"]
    assert sorted(r)==list(range(15))

def test_ssa_bounds():
    r=SalpSwarm(rastrigin,5,-5.12,5.12,population=8,iterations=4,seed=2).run()
    assert np.all(r["best_position"]>=-5.12) and np.all(r["best_position"]<=5.12)

def test_rbf_finite():
    rng=np.random.default_rng(3); X=rng.normal(size=(40,3)); y=X[:,0]-X[:,1]
    m=RBFRegressor(5,1,.001,3).fit(X,y); p=m.predict(X)
    assert p.shape==(40,) and np.isfinite(p).all()

def test_ssa_reproducible():
    a=SalpSwarm(rastrigin,3,-5.12,5.12,8,4,7).run()["best_fitness"]
    b=SalpSwarm(rastrigin,3,-5.12,5.12,8,4,7).run()["best_fitness"]
    assert a==b
