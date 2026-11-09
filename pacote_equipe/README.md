# Ambiente de Avaliação Acadêmica - Offline

Este pacote contém o ambiente de treino e avaliação local para a competição. A aplicação corre dentro de um contentor Docker isolado, garantindo um ambiente idêntico para todos os participantes sem expor o código-fonte proprietário do avaliador.

**Importante:** Este ambiente usa um banco de matrizes **público**, de treino. **Não é o lote oficial** usado para o cálculo da nota final. Ele serve para você testar e refinar o seu algoritmo antes da submissão oficial.

---

## Requisitos

Antes de iniciar, certifique-se de que tem instalado no seu computador:
1. **Docker Desktop** (com o serviço Docker a correr em segundo plano).
2. **Docker Compose** (geralmente já vem incluído no Docker Desktop).

---

## Como Executar a Aplicação

1. **Abra o terminal** (CMD, PowerShell ou Terminal do Linux/macOS) nesta exata pasta.
2. **Inicie o ambiente** executando o comando:
   ```bash
   docker compose up
   ```
   Na primeira vez, o Docker transfere a imagem publicada automaticamente. É o único momento que precisa de internet.
3. **Abra o navegador** em **http://localhost:8501**. Nele, a interface da aplicação irá aparecer.
4. **Para parar**, volte ao terminal e prima `Ctrl+C`, depois execute:
   ```bash
   docker compose down
   ```
   (o serviço está configurado para reiniciar sozinho. `docker compose down` é o comando que efetivamente o desliga.)

---

## Se o `docker compose up` falhar

- Confirme que o Docker Desktop está a correr antes do comando.
- Erro relacionado com `competicao.db`: confira se o passo de preparação
  (`touch competicao.db`) foi feito — tem de ser um ficheiro, não uma pasta.
- Se faltar memória, reduza `mem_limit` no `docker-compose.yml` conforme a
  RAM disponível no seu computador.

---

## O que o seu código precisa ter

O ficheiro `.py` que envia deve conter **uma única função**, assinada exatamente como:

```python
def principal(observada, mascara):
    ...
    return matriz_completa
```

- **`observada`**: A matriz com buracos. Algumas células têm o valor real enquanto outras estão ausentes. São as ausentes que seu algoritmo precisa preencher.
- **`mascara`**: Uma matriz booleana do mesmo tamanho de `observada`, indicando quais células são conhecidas (`True`) e quais estão ocultas (`False`).
- **OBS:** Podem ser usadas funções de suporte, desde que dentro da função `principal`.

### O que a função deve devolver

- Uma matriz com **exatamente as mesmas dimensões** de `observada`, contendo **apenas números** (`int` ou `float`). Nada de texto, `None` ou outros tipos de dados.
- **Sem `NaN` ou `Inf`/`-Inf`** em nenhuma posição.
- O retorno pode ser uma `list` de `list`s ou um `numpy.ndarray` da matriz `observada` completada.
- As posições que já eram conhecidas (`mascara == True`) devem manter o seu valor original. O avaliador não penaliza alterá-las, mas é essa a expectativa: a sua função deve **completar** a matriz, não reescrevê-la.

O avaliador calcula o erro (NRMSE) comparando apenas as posições que estavam ocultas com o valor verdadeiro. É esse número que aparece como resultado de cada teste.