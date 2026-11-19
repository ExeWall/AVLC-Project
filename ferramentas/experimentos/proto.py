import numpy as np

CFG = dict(K=5, reps=1, lam1=0.001, iters_cv=30, iters_final=150, rmax=10, modos=(0, 1, 2),
           lams=(0.0001, 0.001, 0.01, 0.05, 0.2), lam_ef=(0.0, 0.3, 1.0, 3.0), mistura=True, se=0.0, etapa2=True,
           dof=0.9, clip=0.5, debug=False, paciencia=3)
INFO = {}


def principal(observada, mascara, cfg=None):
    c = dict(CFG)
    if cfg:
        c.update(cfg)
    O = np.array(observada, dtype=float)
    M = np.array(mascara) != 0
    if O.ndim != 2:
        return O
    m, n = O.shape
    M = M & np.isfinite(O)
    nk = int(M.sum())
    if nk == 0:
        return np.zeros((m, n))
    X = np.where(M, O, 0.0)
    if nk == m * n:
        return X
    mu = X.sum() / nk
    s = np.sqrt((np.where(M, X - mu, 0.0) ** 2).sum() / nk)
    if not (s > 1e-300):
        return np.where(M, X, mu)
    Z = np.where(M, (X - mu) / s, 0.0)
    Mf = M.astype(float)

    def efeitos(T, W, lam, rodadas=15):
        # T, W: (B, m, n).  devolve a (B,m,1), b (B,1,n)
        cl = W.sum(axis=2, keepdims=True)
        cc = W.sum(axis=1, keepdims=True)
        a = np.zeros(cl.shape)
        b = np.zeros(cc.shape)
        for it in range(rodadas):
            a = (W * (T - b)).sum(axis=2, keepdims=True) / (cl + lam + 1e-12)
            b = (W * (T - a)).sum(axis=1, keepdims=True) / (cc + lam + 1e-12)
        return a, b

    def als(T, W, r, lam, iters, U0=None, V0=None):
        # T, W: (B,m,n); lam: (B,) ; devolve U (B,m,r), V (B,n,r)
        B = T.shape[0]
        WT = W * T
        if U0 is None:
            Us, sv, Vt = np.linalg.svd(WT, full_matrices=False)
            V = np.swapaxes(Vt[:, :r, :], 1, 2) * np.sqrt(sv[:, None, :r])
        else:
            V = V0
        cl = W.sum(axis=2)
        cc = W.sum(axis=1)
        eye = np.eye(r)[None, None, :, :]
        lamb = lam.reshape(B, 1, 1, 1)
        Wt = np.swapaxes(W, 1, 2)
        WTt = np.swapaxes(WT, 1, 2)
        U = None
        for it in range(iters):
            VV = (V[:, :, :, None] * V[:, :, None, :]).reshape(B, n, r * r)
            A = (W @ VV).reshape(B, m, r, r) + lamb * (cl[:, :, None, None] + 1.0) * eye
            U = np.linalg.solve(A, (WT @ V)[:, :, :, None])[:, :, :, 0]
            UU = (U[:, :, :, None] * U[:, :, None, :]).reshape(B, m, r * r)
            A = (Wt @ UU).reshape(B, n, r, r) + lamb * (cc[:, :, None, None] + 1.0) * eye
            V = np.linalg.solve(A, (WTt @ U)[:, :, :, None])[:, :, :, 0]
        return U, V

    # ---------- dobras ----------
    K = c['K']
    reps = c['reps']
    I = np.arange(m)[:, None] * np.ones((1, n))
    J = np.ones((m, 1)) * np.arange(n)[None, :]
    cl0 = M.sum(axis=1, keepdims=True)
    cc0 = M.sum(axis=0, keepdims=True)
    livre = M & (cl0 > 1) & (cc0 > 1)
    Hs = []
    for rep in range(reps):
        h = np.sin(I * 12.9898 + J * 78.233 + rep * 37.719 + 1.0) * 43758.5453
        h = h - np.floor(h)
        if rep % 2 == 0:
            chave = np.where(livre, I + h, 1e9)
        else:
            chave = np.where(livre, J + h, 1e9)
        pos = np.argsort(np.argsort(chave.ravel(), kind='stable'), kind='stable').reshape(m, n)
        fold = (pos + rep) % K
        Hs.append(livre[None, :, :] & (fold[None, :, :] == np.arange(K)[:, None, None]))
    H = np.concatenate(Hs, axis=0)            # (B, m, n) retidas
    B = H.shape[0]
    Hf = H.astype(float)
    W = Mf[None, :, :] * (1.0 - Hf)
    nh = Hf.sum()
    Zb = Z[None, :, :] * np.ones((B, 1, 1))
    if nh < 4:
        a, b = efeitos(Z[None], Mf[None], 1.0)
        est = (a + b)[0]
        return np.where(M, X, mu + s * est)

    def erro(P):
        return ((Hf * (P - Zb)) ** 2).sum() / nh

    cands = []     # (nome, erro, P oof)
    # base: média (zero), efeitos com ridge
    cands.append(('zero', erro(np.zeros((B, m, n))), np.zeros((B, m, n)), None))
    melhor_ef = None
    for lam in c['lam_ef']:
        a, b = efeitos(Zb, W, lam)
        P = a + b
        e = erro(P)
        if melhor_ef is None or e < melhor_ef[1]:
            melhor_ef = ('ef', e, P, lam)
    cands.append(melhor_ef)
    lam_ef = melhor_ef[3]
    Pef = melhor_ef[2]

    # ---------- posto baixo: escolha de posto ----------
    modos = c['modos']
    nm = len(modos)
    rcap = int(min(c['rmax'], min(m, n) - 1))
    tabela = []
    melhor = None
    off = mu / s
    def alvo(md):
        return Zb if md == 0 else (Zb - Pef if md == 1 else (Zb + off) * (W > 0))
    def basef(md):
        return np.zeros((B, m, n)) if md == 0 else (Pef if md == 1 else np.zeros((B, m, n)) - off)
    Tm = np.concatenate([alvo(md) for md in modos], axis=0)
    Wm = np.concatenate([W for md in modos], axis=0)
    base = np.concatenate([basef(md) for md in modos], axis=0)
    ntr = W.sum() / B
    sem_melhora = 0
    for r in range(1, rcap + 1):
        if r * (m + n - r) > c['dof'] * ntr:
            break
        lam = np.full(nm * B, c['lam1'])
        U, V = als(Tm, Wm, r, lam, c['iters_cv'])
        P = base + U @ np.swapaxes(V, 1, 2)
        achou = False
        for q in range(nm):
            Pq = P[q * B:(q + 1) * B]
            e = erro(Pq)
            tabela.append((modos[q], r, e))
            if melhor is None or e < melhor[1] * (1.0 - c['se']):
                melhor = (('lr', modos[q], r, c['lam1']), e, Pq)
                achou = True
        if achou:
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= c['paciencia']:
                break
    if melhor is not None and c['etapa2']:
        nome, e0, P0 = melhor
        md, r = nome[1], nome[2]
        lams = [l for l in c['lams'] if l != c['lam1']]
        nl = len(lams)
        T2 = np.concatenate([alvo(md) for l in lams], axis=0)
        W2 = np.concatenate([W for l in lams], axis=0)
        b2 = basef(md)
        lam = np.repeat(np.array(lams), B)
        U, V = als(T2, W2, r, lam, c['iters_cv'])
        P = U @ np.swapaxes(V, 1, 2)
        for q in range(nl):
            Pq = b2 + P[q * B:(q + 1) * B]
            e = erro(Pq)
            if e < melhor[1]:
                melhor = (('lr', md, r, lams[q]), e, Pq)
    # ---------- ajuste final ----------
    a, b = efeitos(Z[None], Mf[None], lam_ef, 30)
    Fef = (a + b)[0]
    comps_oof = [Pef]
    comps_fin = [Fef]
    if melhor is not None:
        nome, e, Pl = melhor
        md, r, lam = nome[1], nome[2], nome[3]
        T = Z[None] if md == 0 else ((Z - Fef)[None] if md == 1 else ((Z + off) * Mf)[None])
        U, V = als(T, Mf[None], r, np.array([lam]), c['iters_final'])
        Fl = (U @ np.swapaxes(V, 1, 2))[0] + (0.0 if md == 0 else (Fef if md == 1 else -off))
        comps_oof.append(Pl)
        comps_fin.append(Fl)
    # mistura: z ~ w1*ef + w2*lr  (pesos >= 0, soma <= 1) por mínimos quadrados nas retidas
    if c['mistura']:
        hv = H.ravel()
        y = Zb.ravel()[hv]
        Acols = np.stack([p.ravel()[hv] for p in comps_oof], axis=1)
        melhor_w = None
        k = Acols.shape[1]
        # tenta todos os subconjuntos (k <= 2): solução de mínimos quadrados com recorte
        subconj = [[0], [1], [0, 1]] if k == 2 else [[0]]
        melhor_w = (np.zeros(k), (y ** 2).mean())
        for sc in subconj:
            As = Acols[:, sc]
            G = As.T @ As + 1e-9 * np.eye(len(sc))
            w = np.linalg.solve(G, As.T @ y)
            if (w < 0).any() or w.sum() > 1.0 + 1e-9:
                w = np.clip(w, 0.0, 1.0)
                if w.sum() > 1.0:
                    w = w / w.sum()
            e = ((As @ w - y) ** 2).mean()
            if e < melhor_w[1]:
                wf = np.zeros(k)
                for t in range(len(sc)):
                    wf = wf + w[t] * (np.arange(k) == sc[t])
                melhor_w = (wf, e)
        w = melhor_w[0]
        est = sum(w[t] * comps_fin[t] for t in range(k))
    else:
        errs = [cands[0][1], melhor_ef[1]] + ([melhor[1]] if melhor is not None else [])
        fins = [np.zeros((m, n))] + comps_fin
        t = int(np.argmin(np.array(errs)))
        est = fins[t]
        w = t
    if c['debug']:
        INFO['tabela'] = tabela
        INFO['melhor'] = None if melhor is None else (melhor[0], melhor[1])
        INFO['ef'] = (lam_ef, melhor_ef[1])
        INFO['w'] = w
    est = mu + s * est
    lo = X[M].min()
    hi = X[M].max()
    amp = hi - lo
    est = np.clip(est, lo - c['clip'] * amp, hi + c['clip'] * amp)
    est = np.where(np.isfinite(est), est, mu)
    return np.where(M, X, est)
