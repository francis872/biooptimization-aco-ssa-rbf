from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json, os, numpy as np
from urllib.parse import urlparse
from src.aco.aco import AntColonyTSP, nearest_neighbor_tour
from src.ssa.ssa import SalpSwarm, rastrigin
ROOT=Path(__file__).parent

def mongo_status():
    try:
        from src.database.mongodb import MongoRunStore
        s=MongoRunStore(); s.ping(); return {"connected":True,"db":os.getenv("MONGO_DB","biooptimization"),"collection":os.getenv("MONGO_COLLECTION","runs")}
    except Exception as e: return {"connected":False,"error":str(e)}

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self,path):
        p=urlparse(path).path
        if p=="/": return str(ROOT/"templates"/"index.html")
        return str(ROOT/p.lstrip("/"))
    def send_json(self,obj,status=200):
        b=json.dumps(obj,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)).encode()
        self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p=urlparse(self.path).path
        if p=="/api/status": return self.send_json({"ok":True,"mongo":mongo_status()})
        return super().do_GET()
    def do_POST(self):
        n=int(self.headers.get("Content-Length","0")); data=json.loads(self.rfile.read(n) or b"{}"); p=urlparse(self.path).path
        try:
            if p=="/api/ssa":
                dim=int(data.get("dim",10)); r=SalpSwarm(rastrigin,dim,-5.12,5.12,int(data.get("population",30)),int(data.get("iterations",100)),int(data.get("seed",42))).run()
                return self.send_json(r)
            if p=="/api/aco":
                rng=np.random.default_rng(int(data.get("seed",42))); xy=rng.random((int(data.get("cities",20)),2))*100
                d=np.sqrt(((xy[:,None]-xy[None,:])**2).sum(2)); r=AntColonyTSP(d,int(data.get("ants",30)),int(data.get("iterations",100)),float(data.get("alpha",1)),float(data.get("beta",2)),float(data.get("rho",.5)),int(data.get("seed",42))).run()
                nn=nearest_neighbor_tour(d); r["nearest_neighbor_distance"]=nn[1]; return self.send_json(r)
            return self.send_json({"error":"endpoint"},404)
        except Exception as e: return self.send_json({"error":str(e)},500)

if __name__=="__main__":
    print("BioOptimization UI: http://127.0.0.1:5000")
    ThreadingHTTPServer(("0.0.0.0",5000),Handler).serve_forever()
