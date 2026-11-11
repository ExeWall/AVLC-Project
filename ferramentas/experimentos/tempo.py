import sys, json, time, importlib
from rodar import *
mod = importlib.import_module(sys.argv[1]); cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
rng = np.random.default_rng(5); pior = 0; tot = 0; k = 0
for (m, n) in ((40, 40), (40, 30), (32, 32)):
  for r in (2, 6, 12):
    for frac in (0.1, 0.3, 0.5):
        G = rng.normal(size=(m, r)) @ rng.normal(size=(r, n)) + 0.1 * rng.normal(size=(m, n)) + 5
        M = gerar_sinteticas.mascara_aleatoria(rng, m, n, frac)
        t0 = time.perf_counter(); mod.principal(np.where(M, G, np.nan), M.astype(np.int64), cfg) if cfg is not None else mod.principal(np.where(M, G, np.nan), M.astype(np.int64)); dt = time.perf_counter() - t0
        pior = max(pior, dt); tot += dt; k += 1
print('pior=%.3f medio=%.3f' % (pior, tot / k))
