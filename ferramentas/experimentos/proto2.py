import numpy as np

CFG = dict(K=10, reps=1, its0=12, its=5, iters_final=4, rmax=10, modos=(0, 1, 2), bag='final',
           caminho=(0.3, 0.1, 0.03, 0.01, 0.001, 0.0001), lam_ef=(0.0, 0.3, 1.0, 3.0),
           guarda=3.0, dof=0.9, clip=0.25, debug=False, Kad=(500, 800, 1100), mg=700, repmax=99999, tmin=2.0, paciencia=3, pond=True)
INFO = {}


def principal(observada, mascara, cfg=None):
    c = dict(CFG)
    if cfg:
        c.update(cfg)
    O = np.array(observada, dtype=float)
    M = np.array(mascara) != 0
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
    off = mu / s
    zlo = Z[M].min()
    zhi = Z[M].max()
    zamp = zhi - zlo
    zlo = zlo - c['clip'] * zamp
    zhi = zhi + c['clip'] * zamp

    def efeitos(T, W, lam, rodadas=15):
        cl = W.sum(axis=2, keepdims=True)
        cc = W.sum(axis=1, keepdims=True)
        a = np.zeros(cl.shape)
        b = np.zeros(cc.shape)
        for it in range(rodadas):
            a = (W * (T - b)).sum(axis=2, keepdims=True) / (cl + lam + 1e-12)
            b = (W * (T - a)).sum(axis=1, keepdims=True) / (cc + lam + 1e-12)
        return a, b

    def iniciar(WT, r):
        Us, sv, Vt = np.linalg.svd(WT, full_matrices=False)
        return np.swapaxes(Vt[:, :r, :], 1, 2) * np.sqrt(sv[:, None, :r])

    def als(WT, W, V, lam, iters):
        B = WT.shape[0]
        r = V.shape[2]
        eye = np.eye(r)[None, None, :, :]
        if c['pond']:
            pl = lam * (W.sum(axis=2)[:, :, None, None] + 1.0) * eye
            pc = lam * (W.sum(axis=1)[:, :, None, None] + 1.0) * eye
        else:
            pl = lam * eye
            pc = lam * eye
        Wt = np.swapaxes(W, 1, 2)
        WTt = np.swapaxes(WT, 1, 2)
        U = None
        for it in range(iters):
            VV = (V[:, :, :, None] * V[:, :, None, :]).reshape(B, n, r * r)
            U = np.linalg.solve((W @ VV).reshape(B, m, r, r) + pl, (WT @ V)[:, :, :, None])[:, :, :, 0]
            UU = (U[:, :, :, None] * U[:, :, None, :]).reshape(B, m, r * r)
            V = np.linalg.solve((Wt @ UU).reshape(B, n, r, r) + pc, (WTt @ U)[:, :, :, None])[:, :, :, 0]
        return U, V

    # ---------- dobras ----------
    K = c['K']
    if c['Kad']:
        K = 10 if nk <= c['Kad'][0] else (8 if nk <= c['Kad'][1] else (6 if nk <= c['Kad'][2] else 5))
    I = np.arange(m)[:, None] * np.ones((1, n))
    J = np.ones((m, 1)) * np.arange(n)[None, :]
    cl0 = M.sum(axis=1, keepdims=True)
    cc0 = M.sum(axis=0, keepdims=True)
    livre = M & (cl0 > 1) & (cc0 > 1)
    Hs = []
    for rep in range(c['reps'] if nk <= c['repmax'] else 1):
        h = np.sin(I * 12.9898 + J * 78.233 + rep * 37.719 + 1.0) * 43758.5453
        h = h - np.floor(h)
        chave = np.where(livre, (I if rep % 2 == 0 else J) + h, 1e9)
        pos = np.argsort(np.argsort(chave.ravel(), kind='stable'), kind='stable').reshape(m, n)
        fold = (pos + rep) % K
        Hs.append(livre[None, :, :] & (fold[None, :, :] == np.arange(K)[:, None, None]))
    H = np.concatenate(Hs, axis=0)
    B = H.shape[0]
    Hf = H.astype(float)
    W = Mf[None, :, :] * (1.0 - Hf)
    nh = Hf.sum()
    Zb = Z[None, :, :] * np.ones((B, 1, 1))
    a, b = efeitos(Z[None], Mf[None], 1.0, 30)
    if nh < 4:
        return np.where(M, X, mu + s * (a + b)[0])

    def erro(P):
        return ((Hf * (P - Zb)) ** 2).sum() / nh

    melhor_ef = None
    for lam in c['lam_ef']:
        a, b = efeitos(Zb, W, lam)
        P = a + b
        e = erro(P)
        if melhor_ef is None or e < melhor_ef[0]:
            melhor_ef = (e, P, lam)
    e_ef, Pef, lam_ef = melhor_ef
    a, b = efeitos(Z[None], Mf[None], lam_ef, 30)
    Fef = (a + b)[0]

    modos = c['modos'] if nk <= c['mg'] else (1, 2)
    nm = len(modos)
    cam = c['caminho']

    def alvo(md, Zx, Px, Wx):
        return Zx if md == 0 else (Zx - Px * Wx if md == 1 else (Zx + off) * Wx)

    def basef(md, Px):
        return 0.0 * Px if md == 0 else (Px if md == 1 else 0.0 * Px - off)

    WTm = np.concatenate([alvo(md, Zb * W, Pef, W) for md in modos], axis=0)
    Wm = np.concatenate([W for md in modos], axis=0)
    basem = np.concatenate([basef(md, Pef) for md in modos], axis=0)
    ntr = W.sum() / B
    rcap = int(min(c['rmax'], min(m, n) - 1))
    tabela = []
    melhor = None       # (erro, P, md, r, nivel, taus)
    sem_melhora = 0
    for r in range(1, rcap + 1):
        if r * (m + n - r) > c['dof'] * ntr:
            break
        V = iniciar(WTm, r)
        Pant = None
        taus = [[] for q in range(nm)]
        achou = False
        for nv in range(len(cam)):
            U, V = als(WTm, Wm, V, cam[nv], c['its0'] if nv == 0 else c['its'])
            P = np.clip(basem + U @ np.swapaxes(V, 1, 2), zlo, zhi)
            Pg = []
            for q in range(nm):
                Pq = P[q * B:(q + 1) * B]
                if nv > 0 and c['guarda'] > 0:
                    t = c['guarda'] * taus[q][-1]
                    Pa = Pant[q * B:(q + 1) * B]
                    Pq = Pa + np.clip(Pq - Pa, -t, t)
                e = erro(Pq)
                taus[q].append(np.sqrt(e))
                tabela.append((modos[q], r, nv, e))
                if melhor is None or e < melhor[0]:
                    melhor = (e, Pq, modos[q], r, nv, list(taus[q]))
                    achou = True
                Pg.append(Pq)
            Pant = np.concatenate(Pg, axis=0)
        if achou:
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= c['paciencia']:
                break

    comps_oof = [Pef]
    comps_fin = [Fef]
    if melhor is not None:
        e, Pl, md, r, nvb, taus = melhor
        WT = alvo(md, Z[None], Fef[None], Mf[None])
        bs = basef(md, Fef[None])
        V = iniciar(WT, r)
        Fl = None
        for nv in range(nvb + 1):
            U, V = als(WT, Mf[None], V, cam[nv], (c['its0'] if nv == 0 else c['its']) * c['iters_final'])
            P = np.clip(bs + U @ np.swapaxes(V, 1, 2), zlo, zhi)[0]
            if nv > 0 and c['guarda'] > 0:
                t = c['guarda'] * taus[nv - 1]
                P = Fl + np.clip(P - Fl, -t, t)
            Fl = P
        if c['bag'] == 'media':
            Fl = Pl.mean(axis=0)
        elif c['bag'] == 'mediana':
            Fl = np.median(Pl, axis=0)
        elif c['bag'] == 'mix':
            Fl = 0.5 * Fl + 0.5 * Pl.mean(axis=0)
        elif c['bag'] == 'mixmed':
            Fl = 0.5 * Fl + 0.5 * np.median(Pl, axis=0)
        comps_oof.append(Pl)
        comps_fin.append(Fl)
    hv = H.ravel()
    y = Zb.ravel()[hv]
    N = y.shape[0]

    def tstat(ea, eb):
        d = ea - eb
        return d.mean() / (d.std() / np.sqrt(N) + 1e-300)

    def ajusta(cols):
        A = np.stack(cols, axis=1)
        G = A.T @ A + 1e-9 * np.eye(len(cols))
        w = np.linalg.solve(G, A.T @ y)
        if (w < 0).any() or w.sum() > 1.0 + 1e-9:
            w = np.clip(w, 0.0, 1.0)
            if w.sum() > 1.0:
                w = w / w.sum()
        return w, (A @ w - y) ** 2

    tmin = c['tmin']
    pe = Pef.ravel()[hv]
    e_cur = y ** 2
    est = np.zeros((m, n))
    w1, e1 = ajusta([pe])
    wdbg = [0.0, 0.0, 0.0]
    usa_ef = False
    if tstat(e_cur, e1) > tmin:
        e_cur = e1
        est = w1[0] * Fef
        usa_ef = True
        wdbg = [w1[0], 0.0, 0.0]
    if melhor is not None:
        pl = comps_oof[1].ravel()[hv]
        w2, e2 = ajusta([pl])
        w3, e3 = ajusta([pe, pl])
        if e3.mean() < e2.mean():
            cand = (e3, w3[0] * Fef + w3[1] * comps_fin[1], [w3[0], w3[1]])
        else:
            cand = (e2, w2[0] * comps_fin[1], [0.0, w2[0]])
        tt = tstat(e_cur, cand[0])
        if tt > tmin:
            est = cand[1]
            wdbg = cand[2] + [tt]
        else:
            wdbg = wdbg[:2] + [tt]
    w = np.array(wdbg)
    if c['debug']:
        INFO['tabela'] = tabela
        INFO['melhor'] = None if melhor is None else (melhor[2], melhor[3], melhor[4], melhor[0])
        INFO['ef'] = (lam_ef, e_ef)
        INFO['w'] = w
    est = mu + s * np.clip(est, zlo, zhi)
    est = np.where(np.isfinite(est), est, mu)
    return np.where(M, X, est)
