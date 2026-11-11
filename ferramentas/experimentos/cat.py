import sys, json, time
from rodar import *
import baselines, importlib
mod = importlib.import_module(sys.argv[1]); cat = sys.argv[2]; cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
cfg['debug'] = True
for nome, c, g, m, o in gerar_sinteticas.gerar_todas():
    if c != cat: continue
    st, rm, nr, dt = av.avaliar_caso(lambda a, b: mod.principal(a, b, cfg), g, m, o)
    b = [av.avaliar_caso(f, g, m, o)[2] for f in (baselines.media_global, baselines.efeitos)]
    I = mod.INFO
    print('%-22s %-8s oc=%.2f nrmse=%.3f global=%.3f ef=%.3f | %s w=%s %.2fs' % (nome, g.shape, 1-(m!=0).mean(), nr, b[0], b[1], I.get('melhor'), np.round(I.get('w'), 2), dt))
