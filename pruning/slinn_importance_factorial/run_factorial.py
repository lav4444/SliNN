# -*- coding: utf-8 -*-
import os, sys, copy, json, time, statistics

SL = '/home/tomi/code/dipl/slinn'
sys.path.insert(0, SL)
sys.path.insert(0, SL + '/gui')
os.chdir(SL)

import torch
import settings as CFG
CFG.DEV_DATA_SUBSET = int(os.environ.get('ABL_SUBSET', '2000'))

MODEL = '/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt'
DATA = '/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7'
OUT = '/home/tomi/code/dipl/pruning/slinn_importance_factorial'

STEPS = int(os.environ.get('ABL_STEPS', '8'))
STEP_FRAC = float(os.environ.get('ABL_STEP_FRAC', '0.05'))
FT_EPOCHS = int(os.environ.get('ABL_FT_EPOCHS', '2'))
SEEDS = [int(x) for x in os.environ.get('ABL_SEEDS', '0,1,2').split(',')]

CELLS = [
    ('grad_none',   'grad',   'none'),
    ('grad_mean',   'grad',   'mean'),
    ('taylor_none', 'taylor', 'none'),
    ('taylor_mean', 'taylor', 'mean'),
]

import engine as E
import introspect as A
import worker as W

RES = None


def log(msg):
    print(msg, flush=True)
    if RES:
        RES.write(msg + '\n'); RES.flush()


def main():
    global RES
    os.makedirs(OUT, exist_ok=True)
    RES = open(os.path.join(OUT, 'results.txt'), 'w')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {"model_path": MODEL, "dataset_path": DATA, "code_dirs": None, "align_m": 32,
           "metric_tol": 0.95, "phase": "1", "batch": {"train": 8}}

    S = W._setup(cfg, dev)
    model, adapter, ctx = S['model'], S['adapter'], S['ctx']
    teacher, loss_fn = S['teacher'], S['loss_fn']
    mfn, mname, monfn = S['mfn'], S['mname'], S['monfn']
    bs = S['bs']

    g0 = E.gflops(model, adapter, dev)
    p0 = A.count_params(model)
    mon0, full0 = float(monfn(model)), float(mfn(model))

    log('=' * 104)
    log('FAKTORIJELNA ABLACIJA — kriterij vaznosti x normalizacija (SliNN prune petlja)')
    log('Model: yolo26n @640 | dataset: sub10k_open_images_v7 | metrika: {}'.format(mname))
    log('=' * 104)
    log('Uredaj: {} | bs={} | DEV_DATA_SUBSET={}'.format(
        torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', bs, CFG.DEV_DATA_SUBSET))
    log('Korak: {:.1%} originalnih GFLOPs x {} koraka (ukupno {:.0%}) | {} KD epohe po rezu | GROW iskljucen'
        .format(STEP_FRAC, STEPS, STEP_FRAC * STEPS, FT_EPOCHS))
    log('Celije: {} | seedovi: {} (seed = KOJI podskup train podataka)'.format(
        ', '.join(c[0] for c in CELLS), SEEDS))
    log('')
    log('BASELINE  GFLOPs={:.4f}  params={:,}  monitor={:.4f}  puni={:.4f}'.format(g0, p0, mon0, full0))

    summary = {'baseline': {'gflops': g0, 'params': p0, 'monitor': mon0, 'full': full0,
                            'metric': mname, 'steps': STEPS, 'step_frac': STEP_FRAC,
                            'ft_epochs': FT_EPOCHS, 'subset': CFG.DEV_DATA_SUBSET, 'seeds': SEEDS},
               'cells': {}}
    t_all = time.time()

    for seed in SEEDS:
        batches, _src = E.materialize_train_batches(DATA, adapter, dev, ctx['split_plan'], bs, None, seed)
        imp_dev = [E.to_device(b, dev) for b in batches[:3]]
        log('')
        log('#' * 104)
        log('SEED {}  ({} batcheva)'.format(seed, len(batches)))
        log('#' * 104)

        for cname, cmode, cnorm in CELLS:
            CFG.PRUNE_IMPORTANCE, CFG.PRUNE_IMP_NORM = cmode, cnorm
            torch.manual_seed(1234)
            student = copy.deepcopy(model).to(dev)
            st = E.MorphState(set(ctx['prunable']), g0, STEP_FRAC, 0.0, None)
            opt, gstep, warmup = E._new_prodigy(student), 0, len(batches)

            log('')
            log('--- {}  (kriterij={}, norm={}, seed={}) ---'.format(cname.upper(), cmode, cnorm, seed))
            log('    {:>3} {:>9} {:>10} {:>7} {:>8} {:>7}'.format('k', 'GFLOPs', 'params', 'rez', 'monitor', 's'))
            hist, tv0 = [], time.time()
            for step in range(1, STEPS + 1):
                ts = time.time()
                student, mi = E.morph_step(student, teacher, adapter, dev, ctx, st, imp_dev,
                                           loss_fn=loss_fn, grow=False)
                opt = E._new_prodigy(student)
                kd = None
                for _e in range(FT_EPOCHS):
                    kd, gstep = E.kd_epoch(student, teacher, adapter, dev, ctx, batches, None,
                                           opt, gstep, warmup, loss_fn=loss_fn)
                m = float(monfn(student))
                g = E.gflops(student, adapter, dev)
                pn = A.count_params(student)
                hist.append({'step': step, 'gflops': g, 'params': pn, 'removed_ch': mi['n_rem'],
                             'monitor': m, 'kd': kd, 'sec': time.time() - ts})
                log('    {:>3} {:>9.4f} {:>10,} {:>7} {:>8.4f} {:>7.0f}'.format(
                    step, g, pn, mi['n_rem'], m, time.time() - ts))

            gf, fullf = E.gflops(student, adapter, dev), float(mfn(student))
            log('    => GFLOPs {:.4f} ({:.1f}%) | monitor {:.4f} | puni {:.4f} ({:.1f}% baseline) | {:.1f} min'
                .format(gf, 100 * gf / g0, hist[-1]['monitor'], fullf,
                        100 * fullf / max(full0, 1e-9), (time.time() - tv0) / 60.0))
            summary['cells'].setdefault(cname, {})[str(seed)] = {
                'mode': cmode, 'norm': cnorm, 'gflops': gf, 'params': A.count_params(student),
                'monitor': hist[-1]['monitor'], 'full': fullf, 'history': hist}
            json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
            if seed == SEEDS[0]:
                torch.save(student, os.path.join(OUT, '{}_seed{}.pt'.format(cname, seed)))
            del student
            torch.cuda.empty_cache()
        del batches, imp_dev
        torch.cuda.empty_cache()

    log('')
    log('=' * 104)
    log('SAZETAK — puni mAP po celiji (prosjek {} seedova)'.format(len(SEEDS)))
    log('=' * 104)
    log('{:<14}{:>8}{:>7}{:>10}{:>26}{:>11}{:>9}'.format(
        'celija', 'krit', 'norm', 'GFLOPs', 'puni mAP po seedu', 'prosjek', 'sd'))
    log('-' * 104)
    means = {}
    for cname, cmode, cnorm in CELLS:
        rs = summary['cells'].get(cname, {})
        vals = [rs[str(s)]['full'] for s in SEEDS if str(s) in rs]
        if not vals:
            continue
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gm = statistics.mean([rs[str(s)]['gflops'] for s in SEEDS if str(s) in rs])
        means[cname] = mu
        log('{:<14}{:>8}{:>7}{:>10.4f}{:>26}{:>11.4f}{:>9.4f}'.format(
            cname, cmode, cnorm, gm, ' '.join('{:.4f}'.format(v) for v in vals), mu, sd))
    log('-' * 104)
    if len(means) == 4:
        gn, gm_, tn, tm = (means['grad_none'], means['grad_mean'],
                           means['taylor_none'], means['taylor_mean'])
        log('')
        log('RAZLAGANJE EFEKATA (puni mAP, prosjek preko seedova):')
        log('  glavni efekt KRITERIJA    (taylor - grad) = {:+.4f}'.format(((tn + tm) - (gn + gm_)) / 2))
        log('  glavni efekt NORMALIZACIJE (mean - none)  = {:+.4f}'.format(((gm_ + tm) - (gn + tn)) / 2))
        log('  INTERAKCIJA                                = {:+.4f}'.format((tm - tn) - (gm_ - gn)))
        log('  ukupno taylor_mean - grad_none             = {:+.4f}'.format(tm - gn))
    log('')
    log('Ukupno vrijeme: {:.1f} min'.format((time.time() - t_all) / 60.0))
    RES.close()


if __name__ == '__main__':
    main()
