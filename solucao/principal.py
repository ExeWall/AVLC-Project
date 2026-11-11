import numpy as np


def principal(observada, mascara):
    # Primeira versão: média global + efeito de cada linha + efeito de cada coluna.
    O = np.array(observada, dtype=float)
    M = np.array(mascara) != 0
    m, n = O.shape
    if not M.any():
        return np.zeros((m, n))
    X = np.where(M, O, 0.0)
    mu = X.sum() / M.sum()
    cl = M.sum(axis=1)
    cc = M.sum(axis=0)
    a = np.zeros(m)
    b = np.zeros(n)
    for rodada in range(30):
        R = (X - mu - b[None, :]) * M
        a = np.where(cl > 0, R.sum(axis=1) / np.maximum(cl, 1), 0.0)
        R = (X - mu - a[:, None]) * M
        b = np.where(cc > 0, R.sum(axis=0) / np.maximum(cc, 1), 0.0)
    estimativa = mu + a[:, None] + b[None, :]
    saida = np.where(M, X, estimativa)
    return saida
