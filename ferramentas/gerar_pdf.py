"""Gera docs/relatorio.html a partir de docs/relatorio.md (para imprimir em PDF).

Uso: python ferramentas/gerar_pdf.py
Depois abra docs/relatorio.html no navegador e use Imprimir > Salvar como PDF.
(precisa de: pip install markdown)
"""
import os

import markdown

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS = """body{font-family:Georgia,serif;font-size:10.5pt;line-height:1.38;color:#111;max-width:175mm;margin:auto}
h1{font-size:17pt;margin:0 0 6pt} h2{font-size:12.5pt;margin:12pt 0 4pt;border-bottom:1px solid #999;padding-bottom:2pt}
p{margin:4pt 0;text-align:justify} pre{background:#f3f3f3;padding:5pt 8pt;font-size:9.5pt;margin:4pt 0}
code{font-family:Consolas,monospace;font-size:9.5pt}
table{border-collapse:collapse;margin:6pt auto;font-size:9pt} td,th{border:1px solid #888;padding:2pt 6pt;text-align:center}
ul{margin:3pt 0 3pt 16pt;padding:0} li{margin:1pt 0}"""

md = open(os.path.join(RAIZ, "docs", "relatorio.md"), encoding="utf-8").read()
corpo = markdown.markdown(md, extensions=["tables", "fenced_code"])
pagina = '<!DOCTYPE html><html lang="pt-br"><head><meta charset="utf-8"><title>Relatório</title><style>%s</style></head><body>%s</body></html>' % (CSS, corpo)
saida = os.path.join(RAIZ, "docs", "relatorio.html")
open(saida, "w", encoding="utf-8").write(pagina)
print("gerado:", saida)
