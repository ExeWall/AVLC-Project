"""Gerador de matrizes sintéticas para testar a solução longe do banco público.

O lote oficial é outro (e pode ter matrizes de até 40x40), então não basta ir
bem nas 10 matrizes públicas. Aqui criamos vários tipos de matriz, com tamanhos
e porcentagens de entradas ocultas diferentes, sempre do mesmo jeito (semente
fixa) para que os resultados possam ser comparados entre versões do código.

Cada caso é uma tupla (nome, categoria, gabarito, mascara, observada), no mesmo
formato dos arquivos do pacote: mascara int64 (1 = conhecida) e NaN nas ocultas.
"""
import numpy as np

FORMAS = [(5, 5), (8, 8), (6, 9), (14, 5), (12, 12), (16, 16), (18, 22), (10, 30),
          (24, 19), (28, 31), (32, 32), (40, 40), (40, 25), (22, 40), (40, 6), (5, 40)]


def mascara_aleatoria(rng, m, n, frac, garantir=True):
    """Esconde 'frac' das entradas. Com garantir=True, toda linha/coluna fica com >= 1 conhecida."""
    M = np.ones((m, n), dtype=bool)
    pos = [(i, j) for i in range(m) for j in range(n)]
    rng.shuffle(pos)
    faltam = int(round(frac * m * n))
    for i, j in pos:
        if faltam <= 0:
            break
        if garantir and (M[i].sum() <= 1 or M[:, j].sum() <= 1):
            continue
        M[i, j] = False
        faltam -= 1
    return M


def empacotar(nome, cat, G, M):
    G = np.asarray(G, dtype=float)
    O = np.where(M, G, np.nan)
    return (nome, cat, G, M.astype(np.int64), O)


# ---------- tipos de matriz ----------
def posto_baixo(rng, m, n, r, ruido, offset=0.0, escala=1.0):
    U = rng.normal(size=(m, r))
    V = rng.normal(size=(n, r))
    L = U @ V.T
    L = L / (L.std() + 1e-12)
    return offset + escala * (L + ruido * rng.normal(size=(m, n)))


def gerar_todas(semente=2026, completo=True):
    rng = np.random.default_rng(semente)
    casos = []

    def add(cat, G, frac=None, garantir=True):
        m, n = np.shape(G)
        if frac is None:
            frac = rng.uniform(0.10, 0.50)
        M = mascara_aleatoria(rng, m, n, frac, garantir)
        casos.append(empacotar("%s_%03d" % (cat, len(casos)), cat, G, M))

    formas = FORMAS if completo else FORMAS[::3]

    # posto baixo com níveis de ruído variados
    for (m, n) in formas:
        rmax = max(1, min(8, min(m, n) // 3))
        for rep in range(4):
            r = int(rng.integers(1, rmax + 1))
            ruido = float(rng.choice([0.0, 0.01, 0.03, 0.1, 0.3]))
            cat = "posto_baixo_limpo" if ruido <= 0.03 else "posto_baixo_ruido"
            add(cat, posto_baixo(rng, m, n, r, ruido))
        # com deslocamento grande (valores em torno de 50 ou 100, como nas públicas)
        r = int(rng.integers(1, rmax + 1))
        add("posto_baixo_offset", posto_baixo(rng, m, n, r, float(rng.choice([0.02, 0.1])),
                                             offset=float(rng.choice([20, 50, 100])), escala=float(rng.choice([1, 3, 10]))))
        # efeitos de linha/coluna somados a um posto baixo (parecido com notas)
        a = rng.normal(size=(m, 1)) * 2
        b = rng.normal(size=(1, n)) * 2
        add("efeitos_mais_posto", 10 + a + b + posto_baixo(rng, m, n, min(2, rmax), 0.1))

    for (m, n) in formas:
        i = np.arange(m)[:, None]
        j = np.arange(n)[None, :]
        # linhas quase constantes / colunas quase constantes
        add("linhas_constantes", rng.uniform(5, 100, size=(m, 1)) + 0.5 * rng.normal(size=(m, n)))
        add("colunas_constantes", rng.uniform(5, 50, size=(1, n)) + 0.5 * rng.normal(size=(m, n)))
        # soma de efeitos, tabuada, xadrez
        add("soma_efeitos", rng.integers(0, 6, size=(m, 1)) + rng.integers(0, 6, size=(1, n)) + 0.0)
        add("tabuada", (i + 1.0) * (j + 1.0))
        add("xadrez", np.where((i + j) % 2 == 0, 1.0, -1.0))
        # blocos (grupos de linhas e colunas)
        gl = rng.integers(0, 3, size=m)
        gc = rng.integers(0, 3, size=n)
        Gb = rng.integers(1, 10, size=(3, 3)).astype(float)
        add("blocos", Gb[gl][:, gc] + 0.1 * rng.normal(size=(m, n)))
        # sensores: média + amplitude * senoide + ruído
        base = rng.uniform(21, 27, size=(m, 1))
        amp = rng.uniform(2, 6, size=(m, 1))
        add("sensores", base + amp * np.sin(2 * np.pi * (2 * j - 9) / 24) + 0.2 * rng.normal(size=(m, n)))
        # notas de 1 a 5 (arredondadas e cortadas)
        u = rng.uniform(-1, 1, size=(m, 2))
        v = rng.uniform(-1, 1, size=(n, 2))
        notas = 3 + rng.uniform(-0.9, 0.9, size=(m, 1)) + rng.uniform(-1.1, 1.1, size=(1, n)) + 1.2 * (u @ v.T)
        add("notas_1a5", np.clip(np.round(notas), 1, 5))
        # imagem binária de posto baixo
        img = (rng.random((m, 1)) < 0.5) & (rng.random((1, n)) < 0.5) | (rng.random((m, 1)) < 0.2) & (rng.random((1, n)) < 0.6)
        add("imagem_binaria", img.astype(float))
        # sem estrutura: gaussiana pura
        add("sem_estrutura", rng.normal(size=(m, n)) * float(rng.choice([1, 7])))
        # sem estrutura: valores singulares caindo devagar (como a pública 8)
        k = min(m, n)
        Q1, _ = np.linalg.qr(rng.normal(size=(m, k)))
        Q2, _ = np.linalg.qr(rng.normal(size=(n, k)))
        add("espectro_linear", (Q1 * np.linspace(1, 0.05, k)) @ Q2.T * 50)
        # constante + ruído (como a pública 1)
        add("constante_ruido", 50 + 1.7 * rng.normal(size=(m, n)))
        # posto 1 forte + ruído moderado
        add("posto1_ruido", 50 + posto_baixo(rng, m, n, 1, 0.5))

    # ---------- casos de borda ----------
    def borda(nome, G, M):
        casos.append(empacotar("borda_" + nome, "borda", G, np.asarray(M, dtype=bool)))

    G = posto_baixo(rng, 10, 12, 2, 0.05)
    M = mascara_aleatoria(rng, 10, 12, 0.3)
    M2 = M.copy(); M2[:, 4] = False
    borda("coluna_oculta", G, M2)
    M2 = M.copy(); M2[3, :] = False
    borda("linha_oculta", G, M2)
    M2 = M.copy(); M2[3, :] = False; M2[:, 4] = False; M2[:, 7] = False
    borda("linha_e_colunas_ocultas", G, M2)
    borda("todas_conhecidas", G, np.ones_like(M))
    borda("nenhuma_conhecida", G, np.zeros_like(M))
    M2 = np.zeros_like(M); M2[2, 5] = True
    borda("uma_conhecida", G, M2)
    M2 = np.zeros_like(M); M2[2, :] = True
    borda("so_uma_linha_conhecida", G, M2)
    borda("matriz_constante", np.full((9, 7), 4.0), mascara_aleatoria(rng, 9, 7, 0.3))
    borda("zeros", np.zeros((9, 7)), mascara_aleatoria(rng, 9, 7, 0.3))
    borda("valores_enormes", 1e6 * posto_baixo(rng, 12, 12, 2, 0.01, offset=3), mascara_aleatoria(rng, 12, 12, 0.3))
    borda("valores_minusculos", 1e-6 * posto_baixo(rng, 12, 12, 2, 0.01), mascara_aleatoria(rng, 12, 12, 0.3))
    borda("1xn", rng.normal(size=(1, 15)), mascara_aleatoria(rng, 1, 15, 0.3, garantir=False))
    borda("mx1", rng.normal(size=(15, 1)), mascara_aleatoria(rng, 15, 1, 0.3, garantir=False))
    borda("1x1_oculta", np.array([[3.0]]), np.array([[False]]))
    borda("1x1_conhecida", np.array([[3.0]]), np.array([[True]]))
    borda("2x2", np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([[True, True], [True, False]]))
    borda("3x3_pouco", rng.normal(size=(3, 3)), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    borda("muito_oculta", posto_baixo(rng, 20, 20, 2, 0.05), mascara_aleatoria(rng, 20, 20, 0.85))
    borda("esparsa_sem_garantia", posto_baixo(rng, 15, 15, 2, 0.05), mascara_aleatoria(rng, 15, 15, 0.7, garantir=False))
    borda("40x40_metade", posto_baixo(rng, 40, 40, 5, 0.05), mascara_aleatoria(rng, 40, 40, 0.5))
    borda("40x40_cheia", posto_baixo(rng, 40, 40, 12, 0.2), mascara_aleatoria(rng, 40, 40, 0.1))
    borda("outlier", np.where(rng.random((12, 12)) < 0.03, 1e4, 1.0) * posto_baixo(rng, 12, 12, 2, 0.05), mascara_aleatoria(rng, 12, 12, 0.3))
    return casos


if __name__ == "__main__":
    cs = gerar_todas()
    cats = {}
    for c in cs:
        cats[c[1]] = cats.get(c[1], 0) + 1
    print(len(cs), "casos")
    for k in sorted(cats):
        print("  %-22s %d" % (k, cats[k]))
