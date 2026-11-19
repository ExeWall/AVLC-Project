import sys, importlib
from rodar import *
import baselines
f = av.carregar_funcao(sys.argv[1]) if sys.argv[1].endswith('.py') else importlib.import_module(sys.argv[1]).principal
def hard(O, M, k, it=400):
    mu = O[M].mean(); X = np.where(M, O, mu)
    for _ in range(it):
        U, s, Vt = np.linalg.svd(X, full_matrices=False); L = (U[:, :k]*s[:k])@Vt[:k]; X = np.where(M, O, L)
    return X
rows = []
for nome, c, g, m, o in gerar_sinteticas.gerar_todas():
    if c in ('borda', 'sem_estrutura', 'constante_ruido', 'espectro_linear'): continue
    M = m != 0; h = ~M
    nr = av.avaliar_caso(f, g, m, o)[2]
    O0 = np.where(M, g, 0)
    orc = min(av.avaliar_caso(lambda a, b, k=k: hard(O0, M, k), g, m, o)[2] for k in range(1, min(8, min(g.shape))))
    b = min(av.avaliar_caso(fb, g, m, o)[2] for fb in (baselines.media_global, baselines.efeitos, baselines.media_linha, baselines.media_coluna))
    rows.append((nr - min(orc, b), nome, g.shape, h.mean(), nr, orc, b))
rows.sort(reverse=True)
print('soma das perdas vs oraculo: %.3f em %d casos' % (sum(max(r[0], 0) for r in rows), len(rows)))
for r in rows[:22]: print('%-24s %-9s oc=%.2f nosso=%.3f hard_oraculo=%.3f baseline=%.3f perda=%.3f' % (r[1], r[2], r[3], r[4], r[5], r[6], r[0]))
