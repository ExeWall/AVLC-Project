import numpy as np
from rodar import *
def als(T, W, r, lam, iters):
    B, m, n = T.shape; WT = W*T
    Us, sv, Vt = np.linalg.svd(WT, full_matrices=False)
    V = np.swapaxes(Vt[:, :r, :], 1, 2)*np.sqrt(sv[:, None, :r]); cl=W.sum(2); cc=W.sum(1); eye=np.eye(r)
    Wt=np.swapaxes(W,1,2); WTt=np.swapaxes(WT,1,2); outs={}
    for it in range(iters):
        VV=(V[:,:,:,None]*V[:,:,None,:]).reshape(B,n,r*r)
        U=np.linalg.solve((W@VV).reshape(B,m,r,r)+lam*eye,(WT@V)[...,None])[...,0]
        UU=(U[:,:,:,None]*U[:,:,None,:]).reshape(B,m,r*r)
        V=np.linalg.solve((Wt@UU).reshape(B,n,r,r)+lam*eye,(WTt@U)[...,None])[...,0]
        if it+1 in (30,100,300): outs[it+1]=U@np.swapaxes(V,1,2)
    return outs
for k, r in ((9,4),(10,6)):
    nome, cat, g, mk, o = av.carregar_banco_publico()[k-1]
    M = mk != 0; m, n = g.shape
    X = np.where(M, o, 0.0); s = X[M].std(); T = X/s
    K = 10
    rng = np.random.default_rng(0); fold = rng.integers(0, K, size=(m, n))
    H = M[None] & (fold[None] == np.arange(K)[:, None, None]); W = (M[None] & ~H).astype(float)
    for lam in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 0.3):
        outs = als(T[None]*W, W, r, lam, 300)
        for it, P in outs.items():
            e = np.array([np.sqrt(((P[b]-T)[H[b]]**2).mean()) for b in range(K)])
            print(k, 'lam=%g it=%4d' % (lam, it), 'rmse total=%.4f' % np.sqrt(((H*(P-T))**2).sum()/H.sum()), np.round(e, 3))
