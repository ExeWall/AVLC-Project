import sys, json, time, importlib
from rodar import *
import proto2
rng = np.random.default_rng(11)
casos = []
for (m, n) in ((40, 40), (40, 32), (30, 30)):
    for r in (8, 10, 12, 14, 16):
        for frac in (0.1, 0.25, 0.4):
            for ruido in (0.0, 0.05):
                G = rng.normal(size=(m, r)) @ rng.normal(size=(r, n)); G = G / G.std() + ruido * rng.normal(size=(m, n)) + 3
                M = gerar_sinteticas.mascara_aleatoria(rng, m, n, frac)
                casos.append((m, n, r, frac, ruido, G, M))
for arg in sys.argv[1:]:
    cfg = json.loads(arg); res = {}; pior = 0
    for (m, n, r, frac, ruido, G, M) in casos:
        t0 = time.perf_counter(); S = proto2.principal(np.where(M, G, np.nan), M.astype(np.int64), cfg); dt = time.perf_counter() - t0
        h = ~M; nr = np.sqrt(((S - G)[h] ** 2).mean()) / G[h].std()
        res.setdefault(r, []).append(nr); pior = max(pior, dt)
    print(arg, 'pior_t=%.3f' % pior, ' '.join('r%d=%.3f' % (r, np.mean(v)) for r, v in sorted(res.items())))
