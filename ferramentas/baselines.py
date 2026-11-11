"""As cinco estratégias simples do laboratório "Matriz oculta" do site da disciplina.

Servem de régua: a nossa solução só vale a pena se errar menos do que elas.
Todas recebem (observada, mascara) e devolvem a matriz completa.
"""
import numpy as np


def _prep(observada, mascara):
    M = np.asarray(mascara) != 0
    O = np.where(M, np.asarray(observada, dtype=float), 0.0)
    mu = O[M].mean() if M.any() else 0.0
    return O, M, mu


def _medias(O, M, mu, eixo):
    cont = M.sum(axis=eixo)
    soma = O.sum(axis=eixo)
    return np.where(cont > 0, soma / np.maximum(cont, 1), mu)


def media_global(observada, mascara):
    O, M, mu = _prep(observada, mascara)
    return np.where(M, O, mu)


def media_linha(observada, mascara):
    O, M, mu = _prep(observada, mascara)
    return np.where(M, O, _medias(O, M, mu, 1)[:, None])


def media_coluna(observada, mascara):
    O, M, mu = _prep(observada, mascara)
    return np.where(M, O, _medias(O, M, mu, 0)[None, :])


def efeitos(observada, mascara, rodadas=30):
    O, M, mu = _prep(observada, mascara)
    m, n = O.shape
    a = np.zeros(m)
    b = np.zeros(n)
    cl = M.sum(1)
    cc = M.sum(0)
    for _ in range(rodadas):
        R = (O - mu - b[None, :]) * M
        a = np.where(cl > 0, R.sum(1) / np.maximum(cl, 1), 0.0)
        R = (O - mu - a[:, None]) * M
        b = np.where(cc > 0, R.sum(0) / np.maximum(cc, 1), 0.0)
    return np.where(M, O, mu + a[:, None] + b[None, :])


def vizinhas(observada, mascara, q=3):
    O, M, mu = _prep(observada, mascara)
    m, n = O.shape
    ml = _medias(O, M, mu, 1)
    D = (O - ml[:, None]) * M
    S = np.zeros((m, m))
    for i in range(m):
        for r in range(m):
            if i == r:
                continue
            C = M[i] & M[r]
            if C.sum() < 2:
                continue
            u, v = D[i, C], D[r, C]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 0 and nv > 0:
                S[i, r] = u @ v / (nu * nv)
    saida = O.copy()
    for i in range(m):
        for j in range(n):
            if M[i, j]:
                continue
            cand = [(S[i, r], r) for r in range(m) if r != i and M[r, j] and S[i, r] > 0]
            cand.sort(reverse=True)
            cand = cand[:q]
            if not cand:
                saida[i, j] = ml[i]
            else:
                num = sum(s * D[r, j] for s, r in cand)
                den = sum(s for s, r in cand)
                saida[i, j] = ml[i] + num / den
    return saida


TODAS = {
    "global": media_global,
    "linha": media_linha,
    "coluna": media_coluna,
    "efeitos": efeitos,
    "vizinhas": vizinhas,
}
