"""
Torneio de Álgebra Linear (CIN0138) - reconstrução de matrizes com entradas ocultas.

Ideia geral do algoritmo:
  1. Normalizamos a matriz (tiramos a média e dividimos pelo desvio das conhecidas).
  2. Modelo simples: média global + efeito de cada linha + efeito de cada coluna
     (mínimos quadrados alternados com uma pequena regularização).
  3. Modelo principal: aproximação de posto baixo  X ~ U V^T, com U (m x r) e
     V (n x r), ajustada só nas entradas conhecidas por mínimos quadrados
     alternados (ALS): fixando V, cada linha de U é um problema de mínimos
     quadrados pequeno (r incógnitas); depois fixamos U e resolvemos V.
  4. O posto r e a força da regularização NÃO são fixos: escondemos de propósito
     uma parte das entradas conhecidas (validação cruzada), ajustamos o modelo
     com o resto e medimos o erro nas que foram escondidas. Fica a configuração
     que erra menos.
  5. Uma "trava" impede que o modelo menos regularizado se afaste demais do
     modelo mais estável (isso evita previsões absurdas em linhas/colunas com
     poucas entradas conhecidas).
  6. No fim, misturamos o modelo simples e o de posto baixo com pesos calculados
     por mínimos quadrados nas entradas escondidas. Se nenhum modelo mostrar
     ganho real sobre a média, ficamos com a média (matriz sem estrutura).
"""
import numpy as np


def principal(observada, mascara):
    # ------------------------------------------------------------------
    # 0. Leitura dos dados (a máscara chega como 0/1; ocultas chegam como NaN)
    # ------------------------------------------------------------------
    O = np.array(observada, dtype=float)
    M = np.array(mascara) != 0
    if O.ndim != 2 or M.shape != O.shape:
        return np.where(np.isfinite(O), O, 0.0)
    m, n = O.shape
    M = M & np.isfinite(O)                 # só confiamos em valores finitos
    nk = int(M.sum())                      # número de entradas conhecidas
    if nk == 0:
        return np.zeros((m, n))
    X = np.where(M, O, 0.0)                # zeros no lugar dos NaN
    if nk == m * n:
        return X
    mu = X.sum() / nk                      # média das conhecidas
    desvio = np.sqrt((np.where(M, X - mu, 0.0) ** 2).sum() / nk)
    if not (desvio > 1e-300) or not np.isfinite(desvio):
        return np.where(M, X, mu)          # todas as conhecidas são iguais

    # Matriz normalizada: média 0 e desvio 1 nas conhecidas (0 nas ocultas).
    Z = np.where(M, (X - mu) / desvio, 0.0)
    Mf = M.astype(float)
    desloc = mu / desvio                   # usado para trabalhar sem centralizar
    zmin = Z[M].min()
    zmax = Z[M].max()
    folga = 0.25 * (zmax - zmin)
    zmin = zmin - folga                    # limites razoáveis para as estimativas
    zmax = zmax + folga

    # ------------------------------------------------------------------
    # Funções de apoio. Todas trabalham com "lotes": T e W têm formato
    # (B, m, n), ou seja, B problemas resolvidos de uma vez só (um por dobra
    # da validação cruzada). W vale 1 onde a entrada pode ser usada no ajuste.
    # ------------------------------------------------------------------
    def efeitos(T, W, lam, rodadas):
        # Modelo t_ij ~ a_i + b_j. Fixando b, o melhor a_i é a média dos
        # resíduos da linha i (e vice-versa). lam "encolhe" efeitos de linhas
        # ou colunas com poucas entradas conhecidas.
        cont_l = W.sum(axis=2, keepdims=True)
        cont_c = W.sum(axis=1, keepdims=True)
        a = np.zeros(cont_l.shape)
        b = np.zeros(cont_c.shape)
        for rodada in range(rodadas):
            a = (W * (T - b)).sum(axis=2, keepdims=True) / (cont_l + lam + 1e-12)
            b = (W * (T - a)).sum(axis=1, keepdims=True) / (cont_c + lam + 1e-12)
        return a + b

    def iniciar(WT, r):
        # Ponto de partida do ALS: SVD da matriz com zeros nas ocultas.
        # Guardamos as r primeiras direções (vetores singulares à direita).
        Us, sv, Vt = np.linalg.svd(WT, full_matrices=False)
        return np.swapaxes(Vt[:, :r, :], 1, 2) * np.sqrt(sv[:, None, :r])

    def als(WT, W, V, lam, rodadas):
        # Mínimos quadrados alternados para  W * (T - U V^T)  com penalidade lam.
        # Para cada linha i resolvemos as equações normais
        #     (V_i^T V_i + lam * c_i * I) u_i = V_i^T t_i
        # onde V_i só tem as colunas conhecidas na linha i e c_i é quantas são.
        B = WT.shape[0]
        r = V.shape[2]
        ident = np.eye(r)[None, None, :, :]
        pen_l = lam * (W.sum(axis=2)[:, :, None, None] + 1.0) * ident
        pen_c = lam * (W.sum(axis=1)[:, :, None, None] + 1.0) * ident
        Wt = np.swapaxes(W, 1, 2)
        WTt = np.swapaxes(WT, 1, 2)
        U = np.zeros((B, m, r))
        for rodada in range(rodadas):
            VV = (V[:, :, :, None] * V[:, :, None, :]).reshape(B, n, r * r)
            A = (W @ VV).reshape(B, m, r, r) + pen_l
            U = np.linalg.solve(A, (WT @ V)[:, :, :, None])[:, :, :, 0]
            UU = (U[:, :, :, None] * U[:, :, None, :]).reshape(B, m, r * r)
            A = (Wt @ UU).reshape(B, n, r, r) + pen_c
            V = np.linalg.solve(A, (WTt @ U)[:, :, :, None])[:, :, :, 0]
        return U, V

    def alvo(modo, Zx, Ex, Wx):
        # O posto baixo é ajustado em três versões da matriz:
        #   modo 0: matriz centralizada;  modo 1: resíduo depois de tirar os
        #   efeitos de linha/coluna;      modo 2: matriz sem centralizar.
        if modo == 0:
            return Zx * Wx
        if modo == 1:
            return (Zx - Ex) * Wx
        return (Zx + desloc) * Wx

    def base(modo, Ex):
        # O que precisa ser somado de volta à previsão U V^T em cada modo.
        if modo == 0:
            return 0.0 * Ex
        if modo == 1:
            return Ex
        return 0.0 * Ex - desloc

    def completar():
        # --------------------------------------------------------------
        # 1. Dobras da validação cruzada: cada entrada conhecida é escondida
        #    em exatamente uma dobra. O sorteio é feito por uma fórmula fixa
        #    (sem números aleatórios), então o resultado é sempre o mesmo.
        # --------------------------------------------------------------
        K = 10 if nk <= 500 else (8 if nk <= 900 else 6)
        I = np.arange(m)[:, None] * np.ones((1, n))
        J = np.ones((m, 1)) * np.arange(n)[None, :]
        cont_l = M.sum(axis=1, keepdims=True)
        cont_c = M.sum(axis=0, keepdims=True)
        livre = M & (cont_l > 1) & (cont_c > 1)       # única da linha/coluna nunca é escondida
        h = np.sin(I * 12.9898 + J * 78.233 + 1.0) * 43758.5453
        h = h - np.floor(h)                           # número "embaralhado" entre 0 e 1
        chave = np.where(livre, I + h, 1e9)           # ordena linha por linha
        ordem = np.argsort(np.argsort(chave.ravel(), kind="stable"), kind="stable")
        dobra = ordem.reshape(m, n) % K
        H = livre[None, :, :] & (dobra[None, :, :] == np.arange(K)[:, None, None])
        Hf = H.astype(float)                          # 1 nas entradas escondidas da dobra
        W = Mf[None, :, :] * (1.0 - Hf)               # 1 nas entradas usadas no ajuste
        nh = Hf.sum()
        Zb = Z[None, :, :] * np.ones((K, 1, 1))

        Ef = efeitos(Z[None, :, :], Mf[None, :, :], 1.0, 30)[0]
        if nh < 4:
            return Ef                                 # dados demais faltando para validar

        def erro(P):
            # erro quadrático médio nas entradas escondidas (todas as dobras)
            return ((Hf * (P - Zb)) ** 2).sum() / nh

        # --------------------------------------------------------------
        # 2. Modelo de efeitos: escolhe a regularização pela validação
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # 3. Posto baixo: para cada posto r percorremos um "caminho" de
        #    regularização, da mais forte para a mais fraca, aproveitando a
        #    solução anterior como ponto de partida.
        # --------------------------------------------------------------
        modos = (0, 1, 2)
        caminho = (0.3, 0.1, 0.03, 0.01, 0.001, 0.0001)
        WTm = np.concatenate([alvo(q, Zb, Pef, W) for q in modos], axis=0)
        Wm = np.concatenate([W for q in modos], axis=0)
        basem = np.concatenate([base(q, Pef) for q in modos], axis=0)
        n_treino = W.sum() / K
        r_max = int(min(10, min(m, n) - 1))
        achou = False
        melhor_erro = 0.0
        melhor_P = Pef
        melhor_modo = 0
        melhor_r = 0
        melhor_nivel = 0
        melhor_taus = []
        sem_melhora = 0
        for r in range(1, r_max + 1):
            # r(m+n-r) é o número de parâmetros livres de uma matriz de posto r:
            # não faz sentido passar do número de entradas disponíveis.
            if r * (m + n - r) > 0.9 * n_treino:
                break
            V = iniciar(WTm, r)
            P_ant = basem
            taus = [[], [], []]
            melhorou = False
            for nivel in range(len(caminho)):
                U, V = als(WTm, Wm, V, caminho[nivel], 12 if nivel == 0 else 5)
                P = np.clip(basem + U @ np.swapaxes(V, 1, 2), zmin, zmax)
                partes = []
                for q in range(3):
                    Pq = P[q * K:(q + 1) * K]
                    if nivel > 0:
                        # trava: no máximo 3 "erros típicos" de distância do
                        # modelo anterior (mais regularizado e mais estável)
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
                sem_melhora += 1
                if sem_melhora >= 3:
                    break                             # postos maiores só pioram

        # --------------------------------------------------------------
        # 4. Reajuste do melhor posto baixo usando TODAS as conhecidas
        # --------------------------------------------------------------
        Lr = Ef
        if achou:
            WT = alvo(melhor_modo, Z[None, :, :], Ef[None, :, :], Mf[None, :, :])
            soma = base(melhor_modo, Ef[None, :, :])
            V = iniciar(WT, melhor_r)
            for nivel in range(melhor_nivel + 1):
                U, V = als(WT, Mf[None, :, :], V, caminho[nivel], 48 if nivel == 0 else 20)
                P = np.clip(soma + U @ np.swapaxes(V, 1, 2), zmin, zmax)[0]
                if nivel > 0:
                    t = 3.0 * melhor_taus[nivel - 1]
                    P = Lr + np.clip(P - Lr, -t, t)
                Lr = P

        # --------------------------------------------------------------
        # 5. Mistura final. Os pesos saem de um problema de mínimos quadrados
        #    nas entradas escondidas: y ~ w1 * efeitos + w2 * posto_baixo.
        #    Um modelo mais complicado só entra se o ganho for claro
        #    (teste t pareado > 2); senão ficamos com o mais simples.
        # --------------------------------------------------------------
        sel = H.ravel()
        y = Zb.ravel()[sel]
        N = y.shape[0]

        def ganho(e_simples, e_novo):
            d = e_simples - e_novo
            return d.mean() / (d.std() / np.sqrt(N) + 1e-300)

        def pesos(colunas):
            A = np.stack(colunas, axis=1)
            G = A.T @ A + 1e-9 * np.eye(len(colunas))
            w = np.linalg.solve(G, A.T @ y)           # equações normais
            if (w < 0).any() or w.sum() > 1.0:
                w = np.clip(w, 0.0, 1.0)
                if w.sum() > 1.0:
                    w = w / w.sum()
            return w, (A @ w - y) ** 2

        pe = Pef.ravel()[sel]
        e_atual = y ** 2                              # prever a média (zero)
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

    # ------------------------------------------------------------------
    # Execução com rede de segurança: qualquer problema numérico cai para
    # o modelo de efeitos e, em último caso, para a média global.
    # ------------------------------------------------------------------
    try:
        est = completar()
    except Exception:
        try:
            est = efeitos(Z[None, :, :], Mf[None, :, :], 1.0, 30)[0]
        except Exception:
            est = np.zeros((m, n))
    est = np.where(np.isfinite(est), est, 0.0)        # nada de NaN/Inf
    est = mu + desvio * np.clip(est, zmin, zmax)      # volta para a escala original
    est = np.where(np.isfinite(est), est, mu)
    return np.where(M, X, est)                        # conhecidas ficam como estavam
