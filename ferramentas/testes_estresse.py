"""Testes de estresse da solução: tempo no pior caso, tipos de entrada,
casos de borda e repetibilidade.

Uso: python ferramentas/testes_estresse.py solucao/principal.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avaliador_local as av
import gerar_sinteticas as gs


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else "solucao/principal.py"
    f = av.carregar_funcao(caminho)
    falhas = 0

    print("== 1. tempo em matrizes 40x40 (60 casos)")
    rng = np.random.default_rng(7)
    tempos = []
    for k in range(60):
        r = int(rng.integers(1, 15))
        frac = float(rng.choice([0.1, 0.3, 0.5]))
        G = rng.normal(size=(40, r)) @ rng.normal(size=(r, 40)) + float(rng.choice([0, 0.1, 1.0])) * rng.normal(size=(40, 40)) + 5
        M = gs.mascara_aleatoria(rng, 40, 40, frac)
        t0 = time.perf_counter()
        S = np.asarray(f(np.where(M, G, np.nan), M.astype(np.int64)), dtype=float)
        tempos.append(time.perf_counter() - t0)
        if S.shape != G.shape or not np.isfinite(S).all():
            falhas += 1
    print("   pior=%.3fs  médio=%.3fs  (limite do torneio: 2s por matriz)" % (max(tempos), np.mean(tempos)))
    lote = sorted(tempos)[-10:]
    print("   lote com as 10 piores: %.2fs (limite: 25s)" % sum(lote))

    print("== 2. formatos de entrada (máscara int, bool, lista de listas)")
    casos = av.carregar_banco_publico() + gs.gerar_todas(completo=False)
    ref = None
    for tipo in ("int", "bool", "lista"):
        L = av.avaliar(f, casos, tipo_mascara=tipo, detalhe=False)
        ruins = [l for l in L if l[5] is None]
        nr = np.array([l[5] for l in L if l[5] is not None])
        print("   %-5s casos=%d falhas=%d nrmse médio=%.6f" % (tipo, len(L), len(ruins), nr.mean()))
        falhas += len(ruins)
        if ref is None:
            ref = nr
        elif len(nr) != len(ref) or not np.allclose(nr, ref):
            print("   ATENÇÃO: resultado mudou com o tipo da máscara")
            falhas += 1

    print("== 3. casos de borda e todas as sintéticas")
    L = av.avaliar(f, gs.gerar_todas(), detalhe=False)
    ruins = [l for l in L if l[5] is None]
    alter = [l for l in L if l[3] != "ok" and l[5] is not None]
    print("   casos=%d falhas=%d alterou_conhecidas=%d" % (len(L), len(ruins), len(alter)))
    for l in ruins:
        print("   FALHA:", l[0], l[3])
    falhas += len(ruins) + len(alter)

    print("== 4. entradas estranhas")
    estranhos = {
        "observada com inf conhecido": (np.array([[1.0, np.inf, 3.0], [2.0, 4.0, np.nan], [3.0, 6.0, 9.0]]), np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1]])),
        "tudo NaN e máscara 1": (np.full((4, 4), np.nan), np.ones((4, 4), dtype=int)),
        "inteiros": (np.arange(20).reshape(4, 5), (np.arange(20).reshape(4, 5) % 3 != 0).astype(int)),
        "valores 1e12": (1e12 * np.outer(np.arange(1, 9.0), np.arange(1, 9.0)), gs.mascara_aleatoria(rng, 8, 8, 0.3).astype(int)),
        "valores 1e-12": (1e-12 * np.outer(np.arange(1, 9.0), np.arange(1, 9.0)), gs.mascara_aleatoria(rng, 8, 8, 0.3).astype(int)),
        "40x1": (rng.normal(size=(40, 1)), gs.mascara_aleatoria(rng, 40, 1, 0.3, False).astype(int)),
        "1x40": (rng.normal(size=(1, 40)), gs.mascara_aleatoria(rng, 1, 40, 0.3, False).astype(int)),
        "40x2": (rng.normal(size=(40, 2)), gs.mascara_aleatoria(rng, 40, 2, 0.3).astype(int)),
        "linhas repetidas": (np.tile(rng.normal(size=(1, 12)), (12, 1)), gs.mascara_aleatoria(rng, 12, 12, 0.4).astype(int)),
        "metade das colunas ocultas": (rng.normal(size=(10, 10)), (np.arange(10)[None, :] < 5).astype(int) * np.ones((10, 1), dtype=int)),
    }
    for nome, (G, M) in estranhos.items():
        Mb = np.asarray(M) != 0
        obs = np.where(Mb, G, np.nan).astype(float)
        try:
            S = np.asarray(f(obs, np.asarray(M).astype(np.int64)), dtype=float)
            ok = S.shape == obs.shape and np.isfinite(S).all()
        except Exception as e:
            ok = False
            print("   exceção:", repr(e))
        print("   %-30s %s" % (nome, "ok" if ok else "FALHOU"))
        falhas += 0 if ok else 1

    print("== 5. repetibilidade (duas execuções iguais)")
    iguais = True
    for nome, cat, g, m, o in av.carregar_banco_publico():
        a = np.asarray(f(o.copy(), m.copy()), dtype=float)
        b = np.asarray(f(o.copy(), m.copy()), dtype=float)
        iguais = iguais and np.array_equal(a, b)
    print("   ", "ok" if iguais else "RESULTADOS DIFERENTES")
    falhas += 0 if iguais else 1

    print("\nRESULTADO:", "APROVADO" if falhas == 0 else "REPROVADO (%d problemas)" % falhas)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
