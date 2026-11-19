"""Avaliador local: imita o avaliador da monitoria fora do Docker.

Uso:
    python ferramentas/avaliador_local.py solucao/principal.py
    python ferramentas/avaliador_local.py solucao/principal.py --sinteticas
    python ferramentas/avaliador_local.py --baselines

Para cada matriz chama principal(observada, mascara), confere se a saída é
válida (formato, números finitos) e calcula o NRMSE só nas posições ocultas.
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
BANCO = os.path.join(RAIZ, "pacote_equipe", "banco_publico")
sys.path.insert(0, AQUI)


def carregar_funcao(caminho):
    spec = importlib.util.spec_from_file_location("solucao_equipe", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.principal


def carregar_banco_publico():
    casos = []
    for k in range(1, 11):
        g = np.load(os.path.join(BANCO, "gabarito_%02d.npy" % k))
        m = np.load(os.path.join(BANCO, "mascara_%02d.npy" % k))
        o = np.load(os.path.join(BANCO, "observada_%02d.npy" % k))
        casos.append(("publica_%02d" % k, "publica", g, m, o))
    return casos


def normalizador(g_ocultos, g_todos, modo):
    if modo == "std":
        d = g_ocultos.std()
    elif modo == "std_ddof1":
        d = g_ocultos.std(ddof=1) if g_ocultos.size > 1 else 0.0
    elif modo == "std_total":
        d = g_todos.std()
    elif modo == "range":
        d = g_ocultos.max() - g_ocultos.min()
    elif modo == "mean":
        d = abs(g_ocultos.mean())
    else:
        raise ValueError(modo)
    if d < 1e-12:  # como no laboratório: usa a amplitude da matriz
        d = g_todos.max() - g_todos.min()
    return d if d > 1e-12 else 1.0


def avaliar_caso(func, g, m, o, modo="std", tipo_mascara="int"):
    """Devolve (status, rmse, nrmse, tempo)."""
    M = np.asarray(m) != 0
    if tipo_mascara == "bool":
        masc = M.copy()
    elif tipo_mascara == "lista":
        masc = [[int(v) for v in linha] for linha in M]
    else:
        masc = M.astype(np.int64)
    obs = o.copy() if tipo_mascara != "lista" else [list(l) for l in o]
    t0 = time.perf_counter()
    try:
        saida = func(obs, masc)
    except Exception as e:  # o avaliador real desclassifica
        return "EXCECAO: %r" % (e,), None, None, time.perf_counter() - t0
    dt = time.perf_counter() - t0
    try:
        S = np.asarray(saida, dtype=float)
    except Exception:
        return "SAIDA_NAO_NUMERICA", None, None, dt
    if S.shape != g.shape:
        return "FORMATO_ERRADO %s" % (S.shape,), None, None, dt
    if not np.all(np.isfinite(S)):
        return "NAN_OU_INF", None, None, dt
    status = "ok"
    if M.any() and not np.allclose(S[M], g[M], rtol=1e-9, atol=1e-9):
        status = "ok(alterou conhecidas)"
    h = ~M
    if not h.any():
        return status, 0.0, 0.0, dt
    rmse = float(np.sqrt(np.mean((S[h] - g[h]) ** 2)))
    return status, rmse, rmse / normalizador(g[h], g, modo), dt


def avaliar(func, casos, modo="std", tipo_mascara="int", detalhe=True, limite_tempo=1.0):
    linhas = []
    for nome, cat, g, m, o in casos:
        st, rmse, nr, dt = avaliar_caso(func, g, m, o, modo, tipo_mascara)
        linhas.append((nome, cat, g.shape, st, rmse, nr, dt))
        if detalhe:
            aviso = "  <-- LENTO" if dt > limite_tempo else ""
            if nr is None:
                print("%-28s %-9s %s  %.3fs%s" % (nome, str(g.shape), st, dt, aviso))
            else:
                print("%-28s %-9s %-4s rmse=%9.4f nrmse=%7.4f  %.3fs%s" % (nome, str(g.shape), st[:22], rmse, nr, dt, aviso))
    return linhas


def resumo(linhas, por_categoria=False):
    validos = [l for l in linhas if l[5] is not None]
    falhas = [l for l in linhas if l[5] is None]
    nr = np.array([l[5] for l in validos]) if validos else np.array([np.nan])
    tempos = np.array([l[6] for l in linhas])
    print("-" * 78)
    print("casos=%d  falhas=%d  NRMSE medio=%.4f  mediana=%.4f  max=%.4f | tempo total=%.2fs  pior=%.3fs"
          % (len(linhas), len(falhas), nr.mean(), np.median(nr), nr.max(), tempos.sum(), tempos.max()))
    for f in falhas:
        print("  FALHA:", f[0], f[2], f[3])
    if por_categoria:
        cats = sorted(set(l[1] for l in validos))
        for c in cats:
            v = np.array([l[5] for l in validos if l[1] == c])
            print("  %-22s n=%3d  medio=%.4f  max=%.4f" % (c, len(v), v.mean(), v.max()))
    return nr.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", nargs="?", help="arquivo .py com a função principal")
    ap.add_argument("--norm", default="std", choices=["std", "std_ddof1", "std_total", "range", "mean"])
    ap.add_argument("--mascara", default="int", choices=["int", "bool", "lista"])
    ap.add_argument("--sinteticas", action="store_true", help="roda também nas matrizes sintéticas")
    ap.add_argument("--so-sinteticas", action="store_true")
    ap.add_argument("--baselines", action="store_true", help="mostra as 5 estratégias do laboratório")
    ap.add_argument("--quieto", action="store_true")
    a = ap.parse_args()

    publico = carregar_banco_publico()
    if a.baselines:
        import baselines
        for nome, f in baselines.TODAS.items():
            print("\n=== baseline:", nome)
            resumo(avaliar(f, publico, a.norm, a.mascara, detalhe=not a.quieto))
    if a.arquivo:
        func = carregar_funcao(a.arquivo)
        if not a.so_sinteticas:
            print("\n=== banco público:", a.arquivo)
            resumo(avaliar(func, publico, a.norm, a.mascara, detalhe=not a.quieto))
        if a.sinteticas or a.so_sinteticas:
            import gerar_sinteticas
            casos = gerar_sinteticas.gerar_todas()
            print("\n=== sintéticas (%d casos)" % len(casos))
            resumo(avaliar(func, casos, a.norm, a.mascara, detalhe=False), por_categoria=True)


if __name__ == "__main__":
    main()
