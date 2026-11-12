# Relatório – Projeto de Álgebra Linear (CIN0138)

**Equipe 9:** Wallyson Rodrigues da Silva, Guilherme Duque e Bruno Antônio

## Ideia

Começamos pelas estratégias do laboratório do site (médias e efeitos de linha e
coluna). Elas funcionam em algumas matrizes do banco público, mas em outras o
erro fica perto de 1, igual a chutar a média. Olhando os valores singulares dos
gabaritos, vimos que quase todas as matrizes têm posto baixo (entre 1 e 6), ou
seja, podem ser escritas como

    X ≈ U Vᵀ,  com U (m × r) e V (n × r) e r pequeno.

Uma matriz de posto r tem só r(m + n − r) valores livres, então as entradas
conhecidas podem ser suficientes para achar U e V, e o produto devolve as
ocultas.

## Como o código funciona

1. **Normalização.** Tiramos a média e dividimos pelo desvio das entradas
   conhecidas. As ocultas (NaN) viram zero e não entram nas contas.
2. **Efeitos de linha e coluna.** Modelo x_ij ≈ μ + a_i + b_j, ajustado
   alternando a média dos resíduos das linhas e das colunas. É o nosso plano B.
3. **Posto baixo por mínimos quadrados alternados.** Com V fixo, cada linha de
   U é um problema de mínimos quadrados com r incógnitas, resolvido pelas
   equações normais (VᵀV + λI) u = Vᵀx usando só as colunas conhecidas daquela
   linha. Depois fixamos U e resolvemos V, e repetimos. O ponto de partida vem
   da SVD da matriz com zeros nas ocultas. O λ evita que o sistema fique
   instável quando a linha tem poucas entradas conhecidas.
4. **Escolha do posto.** O posto certo muda de matriz para matriz e errar por 1
   já estraga o resultado (na matriz 9, posto 4 dá erro 0,004 e posto 5 dá
   0,43). Como não temos o gabarito, escondemos uma parte das entradas
   conhecidas, ajustamos com o resto e medimos o erro nas que foram escondidas
   (validação cruzada). Testamos postos de 1 a 12 e vários valores de λ e
   ficamos com o de menor erro.
5. **Cuidados.** Com λ muito pequeno algumas previsões saíam absurdas, então
   limitamos o quanto a previsão pode se afastar da versão com λ maior. No fim
   misturamos o posto baixo com o modelo de efeitos, e se nenhum dos dois ganhar
   da média na validação devolvemos a média (caso de matriz sem estrutura).
   Também garantimos que a saída não tenha NaN nem infinito.

## Resultados (banco público, NRMSE)

| Matriz | Tamanho | Efeitos | Vizinhas | Nosso código |
|---|---|---|---|---|
| 1 | 8x8 | 1,264 | 1,310 | 1,234 |
| 2 | 9x6 | 0,033 | 0,043 | 0,034 |
| 3 | 6x9 | 0,077 | 0,371 | 0,076 |
| 4 | 14x5 | 0,303 | 0,231 | 0,041 |
| 5 | 14x7 | 0,453 | 0,268 | 0,162 |
| 6 | 16x16 | 1,115 | 0,906 | 0,042 |
| 7 | 18x22 | 1,054 | 0,752 | 0,088 |
| 8 | 17x17 | 1,066 | 1,208 | 1,003 |
| 9 | 24x19 | 0,248 | 0,266 | 0,001 |
| 10 | 28x31 | 0,327 | 0,335 | 0,001 |
| **Média** | | 0,594 | 0,569 | **0,268** |

"Efeitos" e "Vizinhas" são as duas melhores estratégias do laboratório. Nas
matrizes de posto baixo (4, 6, 7, 9 e 10) o erro cai bastante, e nas 9 e 10 a
reconstrução é quase exata. As matrizes 1 e 8 não têm estrutura: nenhuma
estratégia fica abaixo de 1 e o código devolve a média. Para não depender só
dessas 10 matrizes, testamos também em matrizes geradas por nós, de até 40x40;
o pior tempo foi de cerca de 0,4 s por matriz.

## Limitações

Quando a matriz não tem estrutura, ou quando uma linha ou coluna está toda
oculta, não há informação para recuperar e o resultado fica perto da média. Em
matrizes muito pequenas a validação tem poucas entradas e pode escolher um posto
que não é o melhor.

## Referências

- G. Strang, MIT 18.06, aulas 16 (mínimos quadrados) e 29 (SVD).
- Laboratório "Matriz oculta" e links do site da disciplina (fatoração de matrizes).
- Documentação do NumPy (`numpy.linalg.svd` e `numpy.linalg.solve`).
