import json, sys, time, numpy as np
sys.path.insert(0, "/home/claude/fix/vendor"); sys.path.insert(0, "/home/claude/fix")
from baselines.model import Model
from zhixun_harness.codec import flatten_inputs
P = "/home/claude/fix/cpu-c-guide-20261110-v1"
def load(name):
    X, y, g = [], [], []
    for l in open(f"{P}/public/{name}.jsonl"):
        r = json.loads(l); f = flatten_inputs(r["inputs"], 1)
        X.append([f[k] for k in sorted(f, key=lambda k: (k.split('[')[0], int(k.split('[')[1][:-1])))]); y.append(r["answer"])
        g.append(f'{r["meta"]["source_subset"]}-{r["meta"]["unit"]}')
    return np.array(X, float), np.array(y, float), np.array(g)
xt, yt, gt = load("train"); xv, yv, gv = load("validation")
units = np.unique(gv)
def rmse(m, x, y): return float(np.sqrt(np.mean((m.predict(x).reshape(-1) - y) ** 2)))
results = {}
for seed in range(3):
    rng = np.random.default_rng(seed); half = set(rng.choice(units, len(units)//2, replace=False))
    a = np.array([u in half for u in gv]); xa, ya, xb, yb = xv[a], yv[a], xv[~a], yv[~a]
    parent = Model.load(f"{P}/checkpoint.npz")
    base_a, base_b = rmse(parent, xa, ya), rmse(parent, xb, yb)
    # recent window per unit: last 20% of each train unit
    idx = []
    for u in np.unique(gt):
        w = np.flatnonzero(gt == u); idx.extend(w[int(len(w)*0.8):])
    idx = np.array(sorted(idx))
    for name, rows, epochs in [("recent20", idx, 5), ("full", None, 10)]:
        m = Model.load(f"{P}/checkpoint.npz"); x = xt if rows is None else xt[rows]; y = yt if rows is None else yt[rows]
        best = (base_a, 0, base_b)
        for e in range(1, epochs+1):
            m.fit(x, y, 1); ra = rmse(m, xa, ya); rb = rmse(m, xb, yb)
            results.setdefault((name, e), []).append((ra - base_a, rb - base_b))
            if ra < best[0]: best = (ra, e, rb)
        results.setdefault((name, "selected"), []).append((best[0]-base_a, best[2]-base_b, best[1]))
for k, v in results.items():
    v = np.array(v, dtype=float); print(k, "gate-half Δ %.2f" % v[:,0].mean(), "pseudo-hidden Δ %.2f" % v[:,1].mean(), ("sel-epochs %s" % v[:,2].tolist()) if v.shape[1]>2 else "")
