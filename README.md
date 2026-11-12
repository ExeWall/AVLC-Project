# AVLC-Project

Projeto da disciplina de Álgebra Linear (CIN0138): torneio de reconstrução de
matrizes. A função `principal(observada, mascara)` recebe uma matriz com entradas
ocultas e devolve a matriz completa.

**Equipe 9:** Wallyson Rodrigues da Silva, Guilherme Duque e Bruno Antônio

## Organização

| Pasta | Conteúdo |
|---|---|
| `solucao/principal.py` | **arquivo da entrega** (único, só usa NumPy) |
| `pacote_equipe/` | avaliador da monitoria (Docker) e as 10 matrizes públicas |
| `ferramentas/` | avaliador local, estratégias simples, gerador de matrizes de teste, verificador das restrições, testes de estresse |
| `ferramentas/experimentos/` | protótipos e scripts usados durante o desenvolvimento |
| `docs/` | relatório, roteiro da apresentação e diário de experimentos |
| `entrega/` | zip no formato pedido (`Grupo9.zip`) |

## Como rodar

```bash
pip install numpy scipy RestrictedPython markdown

# resultado nas 10 matrizes públicas (e nas matrizes de teste que geramos)
python ferramentas/avaliador_local.py solucao/principal.py
python ferramentas/avaliador_local.py solucao/principal.py --sinteticas

# estratégias simples do laboratório, para comparar
python ferramentas/avaliador_local.py --baselines --quieto

# confere as restrições do ambiente (sem class, sem semente, etc.)
python ferramentas/checar_restricoes.py solucao/principal.py

# tempo no pior caso, tipos de entrada e casos de borda
python ferramentas/testes_estresse.py solucao/principal.py

# monta o zip da entrega
python ferramentas/montar_entrega.py 9
```

Avaliador oficial: dentro de `pacote_equipe`, `docker compose up -d`, abrir
http://localhost:8501 e enviar `solucao/principal.py`.

## Resultado atual

NRMSE médio no banco público: **0,268** (melhor estratégia simples: 0,569).
Detalhes em `docs/relatorio.md`.
