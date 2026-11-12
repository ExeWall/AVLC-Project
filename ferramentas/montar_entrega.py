"""Monta o .zip da entrega: Grupo9.zip com o código e o relatório.

Uso: python ferramentas/montar_entrega.py 9      (9 = número da equipe)
"""
import os
import sys
import zipfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
numero = sys.argv[1] if len(sys.argv) > 1 else "9"
nome = "Grupo%s" % numero
destino = os.path.join(RAIZ, "entrega", nome + ".zip")
os.makedirs(os.path.dirname(destino), exist_ok=True)
with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(RAIZ, "solucao", "principal.py"), "principal.py")
    z.write(os.path.join(RAIZ, "docs", "relatorio.pdf"), "relatorio.pdf")
print("gerado:", destino)
for info in zipfile.ZipFile(destino).infolist():
    print("  ", info.filename, info.file_size, "bytes")
