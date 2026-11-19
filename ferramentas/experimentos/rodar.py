import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avaliador_local as av, gerar_sinteticas, importlib

def rodar(modnome='proto', cfg=None, pub=True, sint=True, verb=True, casos_s=None):
    mod = importlib.import_module(modnome)
    f = (lambda o, m: mod.principal(o, m, cfg)) if cfg is not None else mod.principal
    out = {}
    if pub:
        L = av.avaliar(f, av.carregar_banco_publico(), detalhe=False)
        nr = [l[5] for l in L]
        out['pub'] = nr
        if verb: print('pub  medio=%.4f | ' % np.mean(nr) + ' '.join('%.3f' % v for v in nr) + ' | t=%.2f pior=%.3f' % (sum(l[6] for l in L), max(l[6] for l in L)))
    if sint:
        cs = casos_s or gerar_sinteticas.gerar_todas()
        L = av.avaliar(f, cs, detalhe=False)
        fal = [l for l in L if l[5] is None]
        cats = {}
        for l in L:
            if l[5] is not None: cats.setdefault(l[1], []).append(l[5])
        nb = [l[5] for l in L if l[5] is not None and l[1] != 'borda']
        out['sint'] = np.mean(nb); out['cats'] = {k: (np.mean(v), np.max(v)) for k, v in cats.items()}
        if verb:
            print('sint medio(sem borda)=%.4f falhas=%d pior_t=%.3f total_t=%.1f' % (np.mean(nb), len(fal), max(l[6] for l in L), sum(l[6] for l in L)))
            print('   ' + ' '.join('%s=%.3f' % (k[:14], np.mean(v)) for k, v in sorted(cats.items())))
        for l in fal: print('FALHA', l[0], l[3])
    return out

if __name__ == '__main__':
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    rodar(sys.argv[1] if len(sys.argv) > 1 else 'proto', cfg)
