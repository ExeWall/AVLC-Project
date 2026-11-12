# Diário de experimentos

Anotações do que fomos testando, na ordem. Os números são NRMSE médio (menor é
melhor): **público** = as 10 matrizes do pacote; **sintético** = 304 matrizes que
geramos (`ferramentas/gerar_sinteticas.py`), sem contar os casos de borda.

| Etapa | O que mudou | Público | Sintético | Observação |
|---|---|---|---|---|
| 0 | Média global | 1,047 | 1,029 | referência (pior caso aceitável) |
| 1 | Efeitos de linha + coluna | 0,594 | 0,798 | primeira versão entregue pelo avaliador local |
| 2 | Posto baixo (ALS) com posto escolhido por validação cruzada, 5 dobras | 0,344 | 0,346 | grande salto; errava o posto nas matrizes 9 e 10 |
| 3 | Ajustar também a matriz **sem centralizar** + regularização menor | 0,335 | 0,344 | as públicas são posto baixo "puro" (sem tirar a média) |
| 4 | 10 dobras em vez de 5 e varrer todos os postos | 0,281 | 0,314 | com 5 dobras sobravam poucos dados para postos altos |
| 5 | Caminho de regularização + "trava" contra previsões que explodem | 0,268 | 0,314 | matrizes 9 e 10 caíram de ~0,1 para 0,001 |
| 6 | Três versões da matriz (centralizada, resíduo dos efeitos, crua) | 0,268 | 0,306 | ajuda nas matrizes com valores deslocados |
| 7 | Menos iterações + parar quando o posto só piora | 0,268 | 0,306 | mesmo erro, 3,6x mais rápido |
| 8 | Número de dobras conforme o tamanho | 0,268 | 0,306 | pior caso 40x40: de 0,75 s para 0,27 s |
| 9 | Posto máximo de 10 para 12 | 0,268 | 0,306 | matrizes grandes de posto 10: 0,30 → 0,19; pior tempo ~0,4 s |

## O que aprendemos em cada etapa

**Etapa 2 – por que posto baixo.** Olhando os valores singulares dos gabaritos
públicos, quase todas as matrizes têm poucos valores singulares grandes (posto 1 a
6) e o resto perto de zero. Só a 1 e a 8 não têm esse formato.

**Etapa 2 – o posto certo é tudo.** Na matriz 9, posto 4 dá erro 0,004 e posto 5
dá 0,43. Não dá para fixar um posto; ele precisa ser escolhido para cada matriz.
Como não temos o gabarito na hora, escondemos parte das entradas conhecidas e
vemos qual posto acerta melhor essas entradas (validação cruzada).

**Etapa 3 – centralizar ou não.** Tirar a média antes aumenta o posto em 1
(matriz de posto 4 vira posto 5 depois de centralizada). Nas matrizes com muitas
ocultas esse posto a mais faz falta. Solução: testar as versões e deixar a
validação decidir.

**Etapa 4 – quantas dobras.** Com 5 dobras o ajuste usa 80% das conhecidas; na
matriz 10 (41% oculta, posto 6) isso não bastava e a validação escolhia um posto
menor do que o certo. Com 10 dobras (90%) passou a acertar.

**Etapa 5 – previsões que explodem.** Com regularização muito pequena, quase
todas as entradas saem perfeitas, mas uma ou outra (em linha/coluna com poucas
conhecidas) sai absurda, porque o sistema linear daquela linha fica mal
condicionado. Com regularização grande nada explode, mas tudo fica "encolhido".
A saída foi usar as duas: seguimos da regularização forte para a fraca e não
deixamos a previsão nova se afastar mais que 3 erros típicos da anterior.
Erro por dobra na matriz 9 (posto 4), antes da trava:
`0,003 0,002 0,002 0,004 0,002 0,130 0,003 0,002 1,099 0,001` – duas dobras
estragavam a média.

**Coisas que testamos e não ajudaram**
- Média dos modelos das dobras (bagging): 0,3065 → 0,3063. Sem ganho, descartado.
- Regularização igual para todas as linhas (sem multiplicar pelo número de
  conhecidas): 0,319 contra 0,314. Pior.
- Limiar do teste de ganho na mistura: 0 → 0,3054; 1 → 0,3061; 2 → 0,3065;
  3 → 0,3315. Entre 0 e 2 tanto faz no geral; ficamos com 2 porque protege as
  matrizes sem estrutura (1,117 contra 1,149), que são as que mais pesam na média.
- Trava com 2 erros típicos em vez de 3: igual.
- Dar mais peso, na validação, às linhas e colunas com poucas conhecidas (onde
  os erros se concentram): 0,311 a 0,343. Piorou, a validação fica mais ruidosa.
- Preferir o posto menor ou o λ maior quando o erro da validação é quase igual:
  0,307 a 0,313. Não ajudou; ficar com o menor erro é melhor.
- Repetir a validação cruzada 2 ou 3 vezes com sorteios diferentes nas matrizes
  pequenas: 0,303 a 0,306 no sintético e 0,270 no público. Diferença dentro do
  ruído, não compensou deixar o código maior.

## Comparação com um "oráculo"

Para saber quanto ainda dava para ganhar, rodamos uma reconstrução por SVD
escolhendo o posto **olhando o gabarito** (o que não é permitido na prática).
Em 256 matrizes sintéticas com estrutura, a nossa solução perde em média só
0,015 para esse oráculo, e a perda se concentra em matrizes minúsculas (5x5,
8x8), onde há poucas entradas para validar. No banco público o oráculo daria
0,259 e nós temos 0,268.

As matrizes 1 e 8 respondem por 2,24 dos 2,68 pontos de erro somados no banco
público. Nelas nenhuma estratégia fica abaixo de 1: os valores ocultos não têm
relação aproveitável com os conhecidos. O melhor que se pode fazer é não piorar,
e por isso o algoritmo recua para a média quando a validação não mostra ganho.

## Outras verificações

- As posições ocultas das matrizes públicas não têm viés (a média das ocultas
  não é sistematicamente maior nem menor que a das conhecidas), então não há o
  que corrigir nesse sentido.

## Restrições do ambiente que cuidamos
- Máscara chega como inteiros 0/1 (não booleana) e as ocultas chegam como NaN.
- Sem `class`, sem semente, sem nomes começando com `_`, funções dentro de `principal`.
- Não escrevemos em posições de arrays (`A[i] = x`): usamos `np.where`, porque
  algumas configurações do RestrictedPython barram esse tipo de escrita.
- Tudo determinístico (as dobras saem de uma fórmula fixa).
