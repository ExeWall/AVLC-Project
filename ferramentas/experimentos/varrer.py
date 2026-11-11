import sys, json
from rodar import *
MOD = 'proto2'
cs = gerar_sinteticas.gerar_todas()
for arg in sys.argv[1:]:
    cfg = json.loads(arg)
    print('###', arg)
    rodar(MOD, cfg, casos_s=cs)
