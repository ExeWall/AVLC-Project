import numpy as np, time
from rodar import *
def als(T, W, r, lam, iters, wr=True):
    m, n = T.shape; WT = W*T
    Us, sv, Vt = np.linalg.svd(WT, full_matrices=False)
    V = Vt[:r].T*np.sqrt(sv[:r]); cl=W.sum(1); cc=W.sum(0); out=[]
    for it in range(iters):
        A = np.einsum('ij,jk,jl->ikl', W, V, V) + lam*(cl[:,None,None]+1 if wr else 1)*np.eye(r)
        U = np.linalg.solve(A, (WT@V)[:,:,None])[:,:,0]
        A = np.einsum('ji,jk,jl->ikl', W, U, U) + lam*(cc[:,None,None]+1 if wr else 1)*np.eye(r)
        V = np.linalg.solve(A, (WT.T@U)[:,:,None])[:,:,0]
        out.append(U@V.T)
    return out
for k, r in ((9,4),(10,6),(6,3),(7,4),(5,3),(4,2)):
    nome, cat, g, m, o = av.carregar_banco_publico()[k-1]
    M = m != 0; h = ~M
    X = np.where(M, o, 0.0); mu = X[M].mean(); s = X[M].std()
    for modo in ('raw','cent'):
        T = np.where(M, (X - (mu if modo=='cent' else 0))/s, 0)
        rr = r if modo=='raw' else r
        for lam in (1e-4, 1e-3, 1e-2, 5e-2):
            outs = als(T, M.astype(float), rr, lam, 400)
            e = [np.sqrt(np.mean(((P*s + (mu if modo=='cent' else 0))[h]-g[h])**2))/g[h].std() for P in outs]
            print(k, modo, 'r=%d lam=%g' % (rr, lam), ' '.join('it%d=%.4f' % (i+1, e[i]) for i in (9, 29, 59, 99, 199, 399)))
