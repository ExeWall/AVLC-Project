# Roteiro da apresentação (Etapa 2) – 10 minutos + 5 de perguntas

**Equipe 9:** Wallyson Rodrigues da Silva, Guilherme Duque e Bruno Antônio

> Combinar antes quem fala cada parte. **Todos** precisam saber responder às
> perguntas do final deste arquivo, porque a monitoria pergunta a cada integrante.

## Divisão do tempo

| Tempo | Parte | Quem | O que mostrar |
|---|---|---|---|
| 0:00–1:00 | Problema e métrica | Wallyson | matriz com buracos; NRMSE = 1 é "chutar a média" |
| 1:00–2:30 | Por onde começamos | Wallyson | médias e efeitos de linha/coluna; erro 0,594 no banco público |
| 2:30–5:00 | Ideia de posto baixo e ALS | Guilherme | valores singulares das públicas; `X ≈ U Vᵀ`; equações normais |
| 5:00–7:00 | Validação cruzada, trava e mistura | Bruno | por que o posto certo importa (matriz 9: posto 4 → 0,004; posto 5 → 0,43) |
| 7:00–8:30 | Demonstração | Bruno | rodar o avaliador e mostrar a tabela |
| 8:30–10:00 | Resultados e o que aprendemos | todos | tabela por matriz; matrizes 1 e 8; limitações |

## Slides sugeridos

1. **Título** – Equipe 9: Wallyson, Guilherme e Bruno.
2. **O problema** – exemplo 4x4 de posto 1 do site (cada linha é múltipla da primeira; as conhecidas determinam as ocultas).
3. **Primeira tentativa** – média global, da linha, da coluna, efeitos. Onde funcionam (matrizes 2 e 3) e onde falham (6 e 7: erro ≈ 1).
4. **O que os dados mostram** – gráfico dos valores singulares de uma matriz pública (ex.: a 9 tem 4 valores grandes e o resto zero). Conclusão: posto baixo.
5. **Posto baixo** – `X ≈ U Vᵀ`; contagem de parâmetros `r(m+n−r)` contra `m·n`.
6. **ALS** – fixar `V`, resolver cada linha por mínimos quadrados; fixar `U`, resolver cada coluna. Mostrar as equações normais.
7. **Qual posto?** – tabela da matriz 9 com erro por posto; validação cruzada: esconder conhecidas de propósito.
8. **Dois cuidados** – regularização + trava (previsões que explodem); recuo para a média quando não há estrutura.
9. **Resultados** – tabela do relatório (0,268 contra 0,569 da melhor estratégia simples).
10. **Interpretação** – 9 e 10 quase exatas; 5 limitada pelo ruído; 1 e 8 sem estrutura.
11. **O que aprendemos** – posto, espaço das colunas, projeção/mínimos quadrados e SVD aplicados a um problema real; a importância de testar fora dos dados públicos.

## Demonstração (ensaiar antes)

```bash
python ferramentas/avaliador_local.py solucao/principal.py          # tabela das 10 públicas
python ferramentas/avaliador_local.py --baselines --quieto          # estratégias simples
python ferramentas/testes_estresse.py solucao/principal.py          # tempo e casos de borda
```

Ou, pelo avaliador oficial: `docker compose up -d` na pasta `pacote_equipe`,
abrir http://localhost:8501 e enviar `solucao/principal.py`.

## Como a equipe evoluiu (para contar na apresentação)

1. Começamos pelas estratégias do laboratório do site (médias e efeitos).
2. Vimos pelos valores singulares que as matrizes tinham posto baixo e fomos estudar SVD (aula 29 do Strang) e fatoração de matrizes.
3. Implementamos o ALS e percebemos que o posto fixo não servia: cada matriz pede um posto.
4. Usamos validação cruzada para escolher o posto; com 5 grupos errava nas matrizes maiores, com 10 passou a acertar.
5. Descobrimos que algumas previsões "explodiam" e criamos a trava.
6. Criamos matrizes de teste próprias para não ficar presos às 10 públicas, e cuidamos do tempo e das restrições do ambiente.

O arquivo `docs/diario_experimentos.md` tem os números de cada etapa.

---

# Perguntas prováveis e respostas

### Conceitos de Álgebra Linear

**1. O que é o posto de uma matriz?**
É a dimensão do espaço das colunas (igual à do espaço das linhas): o número
máximo de colunas linearmente independentes. Posto 1 significa que todas as
linhas são múltiplas de uma só.

**2. Por que posto baixo ajuda a preencher entradas ocultas?**
Porque a matriz passa a ter poucos graus de liberdade: `r(m+n−r)` em vez de
`m·n`. Se as entradas conhecidas são mais numerosas que isso (e bem
espalhadas), elas determinam a matriz inteira.

**3. O que é a SVD?**
Toda matriz pode ser escrita como `A = U Σ Vᵀ`, com `U` e `V` ortogonais e `Σ`
diagonal com os valores singulares em ordem decrescente. O número de valores
singulares não nulos é o posto.

**4. Por que a SVD truncada é a melhor aproximação de posto r?**
É o teorema de Eckart–Young: ficando só com os `r` maiores valores singulares,
obtemos a matriz de posto `r` mais próxima da original (em norma de Frobenius).
O erro é a raiz da soma dos quadrados dos valores singulares descartados.

**5. Onde a SVD aparece no código?**
Na função `iniciar`: é o ponto de partida do ALS (as `r` primeiras direções da
matriz com zeros nas ocultas). Não dá para aplicar a SVD direto e pronto porque
ela precisa da matriz completa, e a nossa tem buracos.

**6. O que são mínimos quadrados e equações normais?**
Quando `Ax = b` não tem solução exata, procuramos o `x` que minimiza
`‖Ax − b‖²`. A solução satisfaz `AᵀA x = Aᵀb` (equações normais), que equivale
a projetar `b` no espaço das colunas de `A`.

**7. Como isso aparece no ALS?**
Com `V` fixo, cada linha `i` vira um problema `V_i u_i ≈ x_i`: as incógnitas são
as `r` coordenadas `u_i` e há uma equação por entrada conhecida da linha.
Resolvemos `(V_iᵀV_i + λ c_i I) u_i = V_iᵀ x_i` com `np.linalg.solve`.

**8. Por que "alternados"?**
O problema em `U` e `V` ao mesmo tempo não é linear (aparece o produto `U Vᵀ`).
Mas fixando um dos dois ele vira mínimos quadrados comum. Então alternamos:
resolve `U`, resolve `V`, e o erro nunca aumenta de um passo para o outro.

**9. O que é a regularização λ e por que existe?**
É o termo `λ I` somado a `VᵀV`. Sem ele, se uma linha tem menos entradas
conhecidas que `r` (ou colunas quase dependentes), o sistema fica singular ou
mal condicionado e a solução explode. Com ele a matriz é sempre invertível e os
coeficientes ficam pequenos quando há pouca informação.

**10. Por que o λ é multiplicado pelo número de conhecidas da linha?**
Para a penalidade acompanhar o tamanho da soma dos erros daquela linha; assim o
mesmo `λ` serve para linhas com muitas e com poucas entradas. Testamos sem isso
e o erro ficou um pouco pior (0,319 contra 0,314).

**11. O que é projeção e qual a relação com o projeto?**
A solução de mínimos quadrados é a projeção de `b` no espaço das colunas de `A`.
Cada passo do ALS projeta os dados de uma linha no espaço gerado pelas direções
em `V` (restrito às colunas conhecidas).

### Sobre as decisões do algoritmo

**12. Por que não usar um posto fixo?**
Porque muda de matriz para matriz (de 1 a 6 nas públicas), e errar por 1 custa
caro: na matriz 9, posto 4 dá 0,004 e posto 5 dá 0,43.

**13. O que é validação cruzada?**
Escondemos de propósito uma parte das entradas conhecidas, ajustamos o modelo
com o resto e medimos o erro nas escondidas. Repetimos para `K` partes, de modo
que cada conhecida é escondida uma vez.

**14. Por que não escolher o posto pelo erro nas próprias entradas usadas no ajuste?**
Porque esse erro sempre diminui quando o posto aumenta (mais parâmetros ajustam
melhor, inclusive o ruído). Escolheríamos sempre o maior posto: é sobreajuste.
O que interessa é o erro em entradas que o modelo não viu.

**15. Vocês usam o gabarito para decidir alguma coisa?**
Não. A função só recebe `observada` e `mascara`. O gabarito público só foi usado
fora do código, para medir o resultado.

**16. Por que 10 grupos e não 5? E por que menos nas matrizes grandes?**
Com 5 grupos o ajuste usa 80% das conhecidas; na matriz 10 isso era pouco para
posto 6 e a validação escolhia posto menor. Com 10, usa 90%. Nas matrizes
grandes há dados de sobra, então menos grupos bastam e o código fica mais rápido.

**17. Como os grupos são formados sem números aleatórios?**
Cada posição `(i, j)` recebe um número "embaralhado" por uma fórmula fixa
(a parte fracionária de um seno), as conhecidas são ordenadas por linha e por
esse número, e o grupo é a posição na ordem módulo `K`. Assim cada linha fica
espalhada entre os grupos e o resultado é sempre igual (não podemos fixar
semente no ambiente do avaliador).

**18. O que é a "trava"?**
Para cada posto ajustamos com regularização forte e depois cada vez mais fraca.
A previsão nova não pode se afastar mais de 3 erros típicos da anterior. Isso
segura as poucas entradas em que o sistema linear fica mal condicionado, sem
perder a precisão nas outras. Foi o que levou as matrizes 9 e 10 de ~0,1 para
0,001.

**19. Por que três versões da matriz (centralizada, resíduo, crua)?**
Tirar a média ou os efeitos pode aumentar o posto (uma matriz de posto 4 vira
posto 5 depois de centralizada). Em outros casos (linha + coluna + pouco mais)
tirar os efeitos reduz o posto. Não sabemos qual é o caso, então a validação
escolhe.

**20. O que acontece numa matriz sem estrutura (como a 8)?**
Nenhum modelo vence a média na validação, o teste de ganho falha e devolvemos a
média das conhecidas. NRMSE ≈ 1, que é o melhor possível quando as ocultas não
dependem das conhecidas.

**21. Por que a matriz 1 dá erro maior que 1 (1,234)?**
Ela é praticamente uma constante mais ruído, 8x8, com só 10 ocultas. A média das
conhecidas difere um pouco da média das ocultas e, com tão poucas ocultas, isso
já passa de 1. Todas as estratégias ficam acima de 1 nela.

**22. O que é o teste de ganho (teste t) na mistura?**
Comparamos, entrada por entrada, o erro quadrático do modelo simples com o do
modelo mais complexo. Se a diferença média for pequena em relação à sua
variação (t < 2), consideramos que o ganho pode ser sorte e ficamos com o
simples.

**23. Como saem os pesos da mistura?**
De outro problema de mínimos quadrados: as colunas de `A` são as previsões de
cada modelo nas entradas escondidas, `y` são os valores verdadeiros dessas
entradas, e resolvemos `AᵀA w = Aᵀy`. Limitamos os pesos entre 0 e 1.

**24. E se uma coluna inteira estiver oculta?**
Não há informação sobre ela na matriz. O efeito da coluna fica zero e o posto
baixo não tem como estimar aquela direção; a previsão vem da média e do efeito
da linha. É uma limitação do problema, não do método.

### Sobre o código e o ambiente

**25. Por que normalizar a matriz?**
Para que os mesmos valores de `λ` e os mesmos limites sirvam para matrizes em
qualquer escala (valores perto de 50, de 0,001 ou de um milhão).

**26. Como garantem que não sai NaN ou infinito?**
As ocultas (NaN) são trocadas por zero logo no início e nunca entram nas contas;
no final trocamos qualquer valor não finito, limitamos ao intervalo dos
conhecidos (com folga) e, se der qualquer erro, devolvemos o modelo de efeitos
ou a média (`try/except`).

**27. Qual o custo computacional?**
Cada passo do ALS monta e resolve `m + n` sistemas `r × r`: custo da ordem de
`m·n·r² + (m+n)·r³`. Multiplica pelo número de passos, de postos testados e de
grupos da validação. Para 40x40 leva no máximo cerca de 0,4 s (limite de 2 s).

**28. Por que o código não tem laços percorrendo entrada por entrada?**
Operações vetorizadas do NumPy são muito mais rápidas, e os `K` ajustes da
validação são resolvidos de uma vez (arrays com uma dimensão a mais). Também
evitamos escrever em posições de arrays (`A[i] = x`) e usamos `np.where`,
porque o ambiente restrito pode barrar esse tipo de escrita.

**29. Por que todas as funções ficam dentro de `principal`?**
É exigência das regras: o avaliador chama só `principal`, e funções de apoio
precisam estar dentro dela. Também não usamos `class` nem semente aleatória.

**30. Como vocês sabem que não ajustaram demais às 10 matrizes públicas?**
Criamos 304 matrizes de teste de tipos e tamanhos variados (até 40x40) e
decidimos as configurações olhando principalmente para elas. O erro médio lá é
0,306, contra 0,642 da melhor estratégia simples.

**31. O que fariam com mais tempo?**
Melhorar as matrizes pequenas e ruidosas (onde a validação é mais incerta), por
exemplo estimando a incerteza de cada entrada pelo condicionamento do sistema
da sua linha e da sua coluna.

**32. Qual foi a maior dificuldade?**
Perceber que o problema não era ajustar o modelo, e sim escolher o posto e
controlar as previsões instáveis. O erro médio era dominado por poucos casos
ruins.
