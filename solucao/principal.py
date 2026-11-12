import numpy as np


def principal(observada, mascara):
    O = np.array(observada, dtype=float)
    M = np.array(mascara) != 0          # a mascara vem como 0/1
    if O.ndim != 2 or M.shape != O.shape:
        return np.where(np.isfinite(O), O, 0.0)
    m, n = O.shape
    M = M & np.isfinite(O)
    nk = int(M.sum())
    if nk == 0:
        return np.zeros((m, n))
    X = np.where(M, O, 0.0)             # tira os NaN das ocultas
    if nk == m * n:
        return X

    # normaliza usando so as conhecidas
    mu = X.sum() / nk
    desvio = np.sqrt((np.where(M, X - mu, 0.0) ** 2).sum() / nk)
    if not (desvio > 1e-300) or not np.isfinite(desvio):
        return np.where(M, X, mu)
    Z = np.where(M, (X - mu) / desvio, 0.0)
    Mf = M.astype(float)
    desloc = mu / desvio
    zmin = Z[M].min()
    zmax = Z[M].max()
    folga = 0.25 * (zmax - zmin)
    zmin = zmin - folga
    zmax = zmax + folga

    # As funcoes abaixo recebem T e W com formato (B, m, n): sao B matrizes
    # resolvidas ao mesmo tempo (uma para cada dobra da validacao).
    # W = 1 nas entradas que podem ser usadas no ajuste.

    def efeitos(T, W, lam, rodadas):
        # t_ij ~ a_i + b_j
        cont_l = W.sum(axis=2, keepdims=True)
        cont_c = W.sum(axis=1, keepdims=True)
        a = np.zeros(cont_l.shape)
        b = np.zeros(cont_c.shape)
        for k in range(rodadas):
            a = (W * (T - b)).sum(axis=2, keepdims=True) / (cont_l + lam + 1e-12)
            b = (W * (T - a)).sum(axis=1, keepdims=True) / (cont_c + lam + 1e-12)
        return a + b

    def iniciar(WT, r):
        # chute inicial de V pela SVD da matriz com zeros nas ocultas
        Us, sv, Vt = np.linalg.svd(WT, full_matrices=False)
        return np.swapaxes(Vt[:, :r, :], 1, 2) * np.sqrt(sv[:, None, :r])

    def als(WT, W, V, lam, rodadas):
        # minimos quadrados alternados: T ~ U V^T so nas entradas com W = 1
        # linha i:  (Vi^T Vi + lam*ci*I) u_i = Vi^T t_i
        B = WT.shape[0]
        r = V.shape[2]
        ident = np.eye(r)[None, None, :, :]
        pen_l = lam * (W.sum(axis=2)[:, :, None, None] + 1.0) * ident
        pen_c = lam * (W.sum(axis=1)[:, :, None, None] + 1.0) * ident
        Wt = np.swapaxes(W, 1, 2)
        WTt = np.swapaxes(WT, 1, 2)
        U = np.zeros((B, m, r))
        for k in range(rodadas):
            VV = (V[:, :, :, None] * V[:, :, None, :]).reshape(B, n, r * r)
            A = (W @ VV).reshape(B, m, r, r) + pen_l
            U = np.linalg.solve(A, (WT @ V)[:, :, :, None])[:, :, :, 0]
            UU = (U[:, :, :, None] * U[:, :, None, :]).reshape(B, m, r * r)
            A = (Wt @ UU).reshape(B, n, r, r) + pen_c
            V = np.linalg.solve(A, (WTt @ U)[:, :, :, None])[:, :, :, 0]
        return U, V

    # modo 0: matriz centralizada, modo 1: residuo dos efeitos, modo 2: sem centralizar
    def alvo(modo, Zx, Ex, Wx):
        if modo == 0:
            return Zx * Wx
        if modo == 1:
            return (Zx - Ex) * Wx
        return (Zx + desloc) * Wx

    def base(modo, Ex):
        if modo == 0:
            return 0.0 * Ex
        if modo == 1:
            return Ex
        return 0.0 * Ex - desloc

    def completar():
        # dobras da validacao (sem sorteio, sempre iguais)
        K = 10 if nk <= 500 else (8 if nk <= 800 else (6 if nk <= 1100 else 5))
        I = np.arange(m)[:, None] * np.ones((1, n))
        J = np.ones((m, 1)) * np.arange(n)[None, :]
        cont_l = M.sum(axis=1, keepdims=True)
        cont_c = M.sum(axis=0, keepdims=True)
        livre = M & (cont_l > 1) & (cont_c > 1)
        h = np.sin(I * 12.9898 + J * 78.233 + 1.0) * 43758.5453
        h = h - np.floor(h)
        chave = np.where(livre, I + h, 1e9)
        ordem = np.argsort(np.argsort(chave.ravel(), kind="stable"), kind="stable")
        dobra = ordem.reshape(m, n) % K
        H = livre[None, :, :] & (dobra[None, :, :] == np.arange(K)[:, None, None])
        Hf = H.astype(float)              # escondidas em cada dobra
        W = Mf[None, :, :] * (1.0 - Hf)   # usadas no ajuste
        nh = Hf.sum()
        Zb = Z[None, :, :] * np.ones((K, 1, 1))

        Ef = efeitos(Z[None, :, :], Mf[None, :, :], 1.0, 30)[0]
        if nh < 4:
            return Ef

        def erro(P):
            return ((Hf * (P - Zb)) ** 2).sum() / nh

        # efeitos de linha e coluna
        melhor_e = -1.0
        lam_ef = 1.0
        Pef = np.zeros((K, m, n))
        for lam in (0.0, 0.3, 1.0, 3.0):
            P = efeitos(Zb * W, W, lam, 15)
            e = erro(P)
            if melhor_e < 0 or e < melhor_e:
                melhor_e = e
                lam_ef = lam
                Pef = P
        Ef = efeitos(Z[None, :, :], Mf[None, :, :], lam_ef, 30)[0]

        # posto baixo: testa os postos e, em cada um, vai diminuindo o lambda
        modos = (0, 1, 2) if nk <= 700 else (1, 2)
        nm = len(modos)
        lams = (0.3, 0.1, 0.03, 0.01, 0.001, 0.0001)
        WTm = np.concatenate([alvo(q, Zb, Pef, W) for q in modos], axis=0)
        Wm = np.concatenate([W for q in modos], axis=0)
        basem = np.concatenate([base(q, Pef) for q in modos], axis=0)
        n_treino = W.sum() / K
        r_max = int(min(12, min(m, n) - 1))
        achou = False
        melhor_erro = 0.0
        melhor_P = Pef
        melhor_modo = 0
        melhor_r = 0
        melhor_nivel = 0
        melhor_taus = []
        sem_melhora = 0
        for r in range(1, r_max + 1):
            if r * (m + n - r) > 0.9 * n_treino:   # mais parametros do que dados
                break
            V = iniciar(WTm, r)
            P_ant = basem
            taus = [[] for q in modos]
            melhorou = False
            for nivel in range(len(lams)):
                U, V = als(WTm, Wm, V, lams[nivel], 12 if nivel == 0 else 5)
                P = np.clip(basem + U @ np.swapaxes(V, 1, 2), zmin, zmax)
                partes = []
                for q in range(nm):
                    Pq = P[q * K:(q + 1) * K]
                    if nivel > 0:
                        # nao deixa fugir muito da previsao com lambda maior
                        t = 3.0 * taus[q][nivel - 1]
                        Pa = P_ant[q * K:(q + 1) * K]
                        Pq = Pa + np.clip(Pq - Pa, -t, t)
                    e = erro(Pq)
                    taus[q].append(np.sqrt(e))
                    if (not achou) or e < melhor_erro:
                        achou = True
                        melhorou = True
                        melhor_erro = e
                        melhor_P = Pq
                        melhor_modo = modos[q]
                        melhor_r = r
                        melhor_nivel = nivel
                        melhor_taus = list(taus[q])
                    partes.append(Pq)
                P_ant = np.concatenate(partes, axis=0)
            if melhorou:
                sem_melhora = 0
            else:
                sem_melhora = sem_melhora + 1
                if sem_melhora >= 3:
                    break

        # refaz o melhor modelo com todas as conhecidas
        Lr = Ef
        if achou:
            WT = alvo(melhor_modo, Z[None, :, :], Ef[None, :, :], Mf[None, :, :])
            soma = base(melhor_modo, Ef[None, :, :])
            V = iniciar(WT, melhor_r)
            for nivel in range(melhor_nivel + 1):
                U, V = als(WT, Mf[None, :, :], V, lams[nivel], 48 if nivel == 0 else 20)
                P = np.clip(soma + U @ np.swapaxes(V, 1, 2), zmin, zmax)[0]
                if nivel > 0:
                    t = 3.0 * melhor_taus[nivel - 1]
                    P = Lr + np.clip(P - Lr, -t, t)
                Lr = P

        # mistura: y ~ w1*efeitos + w2*posto_baixo (pesos por minimos quadrados)
        sel = H.ravel()
        y = Zb.ravel()[sel]
        N = y.shape[0]

        def ganho(e_simples, e_novo):
            d = e_simples - e_novo
            return d.mean() / (d.std() / np.sqrt(N) + 1e-300)

        def pesos(colunas):
            A = np.stack(colunas, axis=1)
            G = A.T @ A + 1e-9 * np.eye(len(colunas))
            w = np.linalg.solve(G, A.T @ y)
            if (w < 0).any() or w.sum() > 1.0:
                w = np.clip(w, 0.0, 1.0)
                if w.sum() > 1.0:
                    w = w / w.sum()
            return w, (A @ w - y) ** 2

        # so troca a media por um modelo se o ganho for claro (t > 2)
        pe = Pef.ravel()[sel]
        e_atual = y ** 2
        est = np.zeros((m, n))
        w1, e1 = pesos([pe])
        if ganho(e_atual, e1) > 2.0:
            e_atual = e1
            est = w1[0] * Ef
        if achou:
            pl = melhor_P.ravel()[sel]
            w2, e2 = pesos([pl])
            w3, e3 = pesos([pe, pl])
            if e3.mean() < e2.mean():
                e_novo = e3
                est_novo = w3[0] * Ef + w3[1] * Lr
            else:
                e_novo = e2
                est_novo = w2[0] * Lr
            if ganho(e_atual, e_novo) > 2.0:
                est = est_novo
        return est

    # se der qualquer problema numerico, cai para o modelo simples
    try:
        est = completar()
    except Exception:
        try:
            est = efeitos(Z[None, :, :], Mf[None, :, :], 1.0, 30)[0]
        except Exception:
            est = np.zeros((m, n))
    est = np.where(np.isfinite(est), est, 0.0)
    est = mu + desvio * np.clip(est, zmin, zmax)
    est = np.where(np.isfinite(est), est, mu)
    return np.where(M, X, est)
