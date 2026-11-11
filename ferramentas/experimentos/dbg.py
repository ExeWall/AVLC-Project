import sys, json, numpy as np
from rodar import *
import proto2 as proto
cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
cfg['debug'] = True
for nome, cat, g, m, o in av.carregar_banco_publico():
    st, rm, nr, dt = av.avaliar_caso(lambda a, b: proto.principal(a, b, cfg), g, m, o)
    I = proto.INFO
    tab = ''
    print(nome, g.shape, 'nrmse=%.3f' % nr, 'melhor=', I['melhor'], 'ef=(%.2f,%.3f)' % I['ef'], 'w=', np.round(I['w'], 2), '%.2fs' % dt)
    print('    ', tab)
