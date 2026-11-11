"""Confere se o arquivo de entrega respeita as restrições do avaliador.

1) Verificação estática (ast): sem class, sem semente, sem nomes com "_" na
   frente, só bibliotecas permitidas, tudo dentro de principal(observada, mascara).
2) Se o pacote RestrictedPython estiver instalado, compila e executa o código
   dentro dele (com guardas bem rígidas) no banco público, para aproximar o
   ambiente do avaliador e medir o tempo.

Uso: python ferramentas/checar_restricoes.py solucao/principal.py
"""
import ast
import os
import sys
import time

import numpy as np

PERMITIDAS = {"numpy", "scipy", "math", "random"}
PROIBIDOS = {"seed", "default_rng", "RandomState", "Random", "print", "open", "exec", "eval",
             "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input",
             "type", "dir", "breakpoint", "memoryview", "super", "object", "time"}


def checar_estatico(caminho):
    fonte = open(caminho, encoding="utf-8").read()
    arv = ast.parse(fonte)
    erros = []
    avisos = []

    def err(no, msg):
        erros.append("linha %d: %s" % (getattr(no, "lineno", 0), msg))

    tops = [n for n in arv.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    funcs = [n for n in tops if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or funcs[0].name != "principal":
        erros.append("deve existir exatamente uma função de topo chamada principal")
    else:
        args = [a.arg for a in funcs[0].args.args]
        if args != ["observada", "mascara"]:
            erros.append("assinatura deve ser principal(observada, mascara)")
    for n in tops:
        if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            err(n, "só imports e a função principal podem ficar no topo do arquivo")

    for no in ast.walk(arv):
        if isinstance(no, ast.ClassDef):
            err(no, "class é proibido")
        elif isinstance(no, ast.Import):
            for a in no.names:
                if a.name.split(".")[0] not in PERMITIDAS:
                    err(no, "import não permitido: " + a.name)
        elif isinstance(no, ast.ImportFrom):
            if (no.module or "").split(".")[0] not in PERMITIDAS:
                err(no, "import não permitido: %s" % no.module)
            for a in no.names:
                if a.name in PROIBIDOS:
                    err(no, "import proibido: " + a.name)
        elif isinstance(no, ast.Name):
            if no.id.startswith("_"):
                err(no, "nome começando com _: " + no.id)
            if no.id in PROIBIDOS:
                err(no, "nome proibido: " + no.id)
        elif isinstance(no, ast.Attribute):
            if no.attr.startswith("_"):
                err(no, "atributo começando com _: " + no.attr)
            if no.attr in ("seed", "default_rng", "RandomState", "Random"):
                err(no, "atributo proibido: " + no.attr)
        elif isinstance(no, ast.arg):
            if no.arg.startswith("_"):
                err(no, "argumento começando com _: " + no.arg)
        elif isinstance(no, ast.FunctionDef):
            if no.name.startswith("_"):
                err(no, "função começando com _: " + no.name)
        elif isinstance(no, ast.AugAssign):
            if isinstance(no.target, (ast.Subscript, ast.Attribute)):
                err(no, "atribuição aumentada em item/atributo (use A[i] = A[i] + x)")
        elif isinstance(no, (ast.Global, ast.Nonlocal)):
            err(no, "global/nonlocal é proibido")
        elif isinstance(no, ast.While):
            avisos.append("linha %d: while (cuidado com laço infinito)" % no.lineno)
        elif isinstance(no, (ast.Assign, ast.AnnAssign)):
            alvos = no.targets if isinstance(no, ast.Assign) else [no.target]
            for alvo in alvos:
                for sub in ast.walk(alvo):
                    if isinstance(sub, ast.Subscript):
                        avisos.append("linha %d: escrita em item (A[i] = x); pode ser barrada por guardas rígidas" % no.lineno)
        elif isinstance(no, (ast.Yield, ast.YieldFrom, ast.Await, ast.AsyncFunctionDef, ast.With, ast.Delete)):
            err(no, "construção não recomendada no ambiente restrito: " + type(no).__name__)
    return fonte, erros, avisos


def carregar_restrito(fonte):
    """Compila com RestrictedPython e devolve a função principal (ou None se não instalado)."""
    try:
        from RestrictedPython import compile_restricted, safe_builtins
        from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
        from RestrictedPython.Guards import (full_write_guard, guarded_iter_unpack_sequence,
                                             guarded_unpack_sequence, safer_getattr)
    except ImportError:
        return None
    import math, random, operator
    import scipy

    def importar(nome, *resto):
        if nome.split(".")[0] not in PERMITIDAS:
            raise ImportError("import bloqueado: " + nome)
        return __import__(nome, *resto)

    def getattr_guardado(obj, nome, *padrao):
        if nome in ("seed", "default_rng", "RandomState", "Random"):
            raise AttributeError("acesso proibido: " + nome)
        return safer_getattr(obj, nome, *padrao)

    ops = {"+=": operator.iadd, "-=": operator.isub, "*=": operator.imul, "/=": operator.itruediv,
           "//=": operator.ifloordiv, "%=": operator.imod, "**=": operator.ipow, "<<=": operator.ilshift,
           ">>=": operator.irshift, "&=": operator.iand, "^=": operator.ixor, "|=": operator.ior,
           "@=": operator.imatmul}
    builtins = dict(safe_builtins)
    for nome, f in (("__import__", importar), ("sum", sum), ("min", min), ("max", max), ("map", map),
                    ("enumerate", enumerate), ("list", list), ("dict", dict), ("set", set),
                    ("any", any), ("all", all), ("reversed", reversed), ("filter", filter)):
        builtins[nome] = f
    g = {
        "__builtins__": builtins,
        "__name__": "solucao_restrita",
        "_getattr_": getattr_guardado,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_": full_write_guard,   # rígida: barra escrita em item de ndarray
        "_inplacevar_": lambda op, a, b: ops[op](a, b),
    }
    codigo = compile_restricted(fonte, "<entrega>", "exec")
    exec(codigo, g)
    return g["principal"]


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else "solucao/principal.py"
    fonte, erros, avisos = checar_estatico(caminho)
    print("== verificação estática:", caminho)
    for a in avisos:
        print("  aviso:", a)
    for e in erros:
        print("  ERRO:", e)
    if not erros:
        print("  ok (%d linhas)" % len(fonte.splitlines()))

    func = None
    try:
        func = carregar_restrito(fonte)
    except Exception as e:
        erros.append("RestrictedPython recusou o código: %r" % (e,))
        print("  ERRO:", erros[-1])
    if func is None and not erros:
        print("== RestrictedPython não instalado; pulei a execução restrita")
    elif func is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import avaliador_local as av
        import gerar_sinteticas
        print("== execução dentro do RestrictedPython (banco público)")
        linhas = av.avaliar(func, av.carregar_banco_publico(), detalhe=False)
        av.resumo(linhas)
        if any(l[5] is None for l in linhas):
            erros.append("falha na execução restrita")
        print("== execução restrita nas sintéticas")
        linhas = av.avaliar(func, gerar_sinteticas.gerar_todas(), detalhe=False)
        av.resumo(linhas)
        if any(l[5] is None for l in linhas):
            erros.append("falha na execução restrita (sintéticas)")
        rng = np.random.default_rng(1)
        pior = 0.0
        for frac in (0.1, 0.3, 0.5):
            G = rng.normal(size=(40, 5)) @ rng.normal(size=(5, 40)) + 0.1 * rng.normal(size=(40, 40))
            M = gerar_sinteticas.mascara_aleatoria(rng, 40, 40, frac)
            t0 = time.perf_counter()
            func(np.where(M, G, np.nan), M.astype(np.int64))
            pior = max(pior, time.perf_counter() - t0)
        print("== pior tempo 40x40 dentro do RestrictedPython: %.3fs" % pior)
    print("\nRESULTADO:", "REPROVADO" if erros else "APROVADO")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
