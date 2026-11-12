"""Independent causal, direct 24-h LightGBM experiment. No pilot code imports."""
import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs' / 'challenge_v1'
TZ = 'Asia/Shanghai'
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
RECIPES = {
    'local_l1': ('local', 'regression_l1', False),
    'spatial_l1': ('spatial', 'regression_l1', False),
    'spatial_l2': ('spatial', 'regression', False),
    'spatial_delta_l1': ('spatial', 'regression_l1', True),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def stamp():
    return datetime.now().astimezone().isoformat()


def code_hashes():
    return {name: sha(ROOT / name) for name in ['pipeline.py', 'test_contract.py']} | {
        'config.json': sha(ROOT / 'config.json'), 'PROTOCOL.md': sha(ROOT / 'PROTOCOL.md')}


def snapshot():
    if (ROOT / 'snapshot_manifest.json').exists():
        raise RuntimeError('Snapshot exists; verify it, do not overwrite.')
    project = ROOT.parents[1]
    data = project / 'data/air_quality/beijing'
    pilot = project / 'experiments/changping_pm25_pilot'
    run = pilot / 'runs/pilot_20261026'
    pairs = [(data / f'processed/observations_{y}.parquet', ROOT / f'data_snapshot/observations_{y}.parquet') for y in range(2021, 2026)]
    pairs += [(data / f'quality/{name}', ROOT / f'data_snapshot/{name}') for name in ['station_registry.csv', 'download_manifest.csv', 'data_quality.json']]
    pairs += [(run / f'predictions_{split}_{m}.parquet', ROOT / f'reference_snapshot/predictions_{split}_{m}.parquet')
              for split in ['validation'] for m in ['tree', 'lstm', 'persistence', 'daily_repeat']]
    pairs += [(run / name, ROOT / f'reference_snapshot/{name}') for name in ['report.md', 'model_selection.json', 'manifest.json', 'model_stages.json']]
    pairs += [(pilot / 'config.json', ROOT / 'reference_snapshot/config.json')]
    records = []
    for src, dst in pairs:
        dst.parent.mkdir(exist_ok=True, parents=True)
        before = sha(src)
        shutil.copy2(src, dst)
        assert before == sha(src) == sha(dst), f'Source changed during copy: {src}'
        records.append({'source': str(src), 'snapshot': str(dst.relative_to(ROOT)), 'sha256': before, 'bytes': dst.stat().st_size})
    chunks = []
    for year in range(2021, 2026):
        frame = pd.read_parquet(ROOT / f'data_snapshot/observations_{year}.parquet')
        part = frame[frame.source_station_name.eq(CFG['target']) & frame.station_version.eq('new_network')]
        chunks.append(part.set_index('hour_start')[['pm25', 'pm10']])
    target = pd.concat(chunks).sort_index()
    assert not target.index.has_duplicates
    target = target.reindex(pd.date_range('2017-01-01', CFG['reserved_from'], inclusive='left', freq='h', tz=TZ))
    target.index.name = 'hour_start'
    dst = ROOT / 'data_snapshot/target.parquet'
    target.to_parquet(dst)
    records.append({'source': 'derived exclusively from observations_2021..2025 current-name target rows; complete axis',
                    'snapshot': str(dst.relative_to(ROOT)), 'sha256': sha(dst), 'bytes': dst.stat().st_size})
    dump(ROOT / 'snapshot_manifest.json', {'created_at': stamp(), 'reference_thread': '01a1762a-858e-7900-bcc3-be8b15c48d2b',
                                          'reserved_from': CFG['reserved_from'], 'files': records})
    print(f'Snapshot complete: {len(records)} files', flush=True)


def verify_snapshot():
    manifest = json.loads((ROOT / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        assert sha(ROOT / row['snapshot']) == row['sha256'], row['snapshot']
    return manifest


def read_data(last_year=2025, stations=None):
    """No 2026 values loaded at development stage."""
    if last_year >= 2026:
        raise ValueError('2026 data is reserved by the user; this pipeline cannot open it.')
    yframe = pd.read_parquet(ROOT / 'data_snapshot/target.parquet')
    yframe = yframe.loc[yframe.index < pd.Timestamp(f'{last_year + 1}-01-01', tz=TZ)]
    frames = []
    for year in range(2021, last_year + 1):
        f = pd.read_parquet(ROOT / f'data_snapshot/observations_{year}.parquet',
                            columns=['hour_start', 'source_station_name', 'station_version', 'pm25', 'pm10'])
        frames.append(f[f.station_version.eq('new_network')])
    frame = pd.concat(frames, ignore_index=True)
    assert not frame.duplicated(['hour_start', 'source_station_name']).any()
    panel = frame.pivot(index='hour_start', columns='source_station_name', values='pm25').reindex(yframe.index)
    if stations is None:
        fit = panel.loc[(panel.index >= pd.Timestamp(CFG['training_start'], tz=TZ)) & (panel.index < pd.Timestamp(CFG['validation_start'], tz=TZ))]
        coverage = fit.notna().mean()
        stations = sorted(coverage[coverage >= .8].index.tolist())
        dump(RUN / 'stations.json', {'selected': stations, 'coverage_training_only': coverage.to_dict(), 'cutoff_exclusive': CFG['validation_start']})
    panel = panel.reindex(columns=stations)
    return yframe.pm25.astype('float32'), yframe.pm10.astype('float32'), panel.astype('float32'), stations


def eligibility(y):
    return (np.arange(len(y)) >= 168) & y.rolling(24, min_periods=24).count().eq(24).to_numpy()


def positions(y, left, right):
    issue = y.index + pd.Timedelta(hours=1)
    return np.flatnonzero(eligibility(y) & (issue >= pd.Timestamp(left, tz=TZ)) &
                         (issue + pd.Timedelta(hours=24) <= pd.Timestamp(right, tz=TZ)))


def labels(y, pos):
    return np.column_stack([y.shift(-h).to_numpy()[pos] for h in range(1, 25)])


def feature_frames(y, pm10, panel):
    """Only nonnegative lags, past rolling windows and known calendar values."""
    d = {}
    for lag in list(range(0, 13)) + [18, 24, 36, 48, 72, 96, 168]:
        d[f'local_lag_{lag}'] = y.shift(lag)
    for w in [3, 6, 12, 24, 48, 72, 168]:
        r = y.rolling(w, min_periods=1)
        for name, s in [('mean', r.mean()), ('std', r.std(ddof=0)), ('min', r.min()), ('max', r.max()), ('count', r.count())]:
            d[f'local_{name}_{w}'] = s
    for lag in [1, 3, 6, 12, 24]:
        d[f'local_delta_{lag}'] = y - y.shift(lag)
    local = pd.DataFrame(d, index=y.index).astype('float32')
    for lag in [0, 1, 3, 6, 12, 24]:
        d[f'pm10_lag_{lag}'] = pm10.shift(lag)
    d['pm10_minus_pm25'] = pm10 - y
    d['pm25_pm10_ratio'] = y / pm10.clip(lower=1)
    for i, name in enumerate(panel.columns):
        if name == CFG['target']:
            continue
        s = panel[name]
        for lag in [0, 1, 3, 6, 12, 24]:
            d[f's{i}_lag_{lag}'] = s.shift(lag)
        d[f's{i}_mean_6'] = s.rolling(6, min_periods=1).mean()
        d[f's{i}_mean_24'] = s.rolling(24, min_periods=1).mean()
        d[f's{i}_delta_3'] = s - s.shift(3)
        d[f's{i}_minus_local'] = s - y
    peers = panel.drop(columns=[CFG['target']], errors='ignore')
    for key, s in {'mean': peers.mean(axis=1), 'median': peers.median(axis=1),
                   'max': peers.max(axis=1), 'min': peers.min(axis=1), 'std': peers.std(axis=1),
                   'count': peers.notna().sum(axis=1)}.items():
        d[f'network_{key}'] = s
        if key != 'count':
            d[f'network_{key}_minus_local'] = s - y
            d[f'network_{key}_delta_3'] = s - s.shift(3)
    return {'local': local, 'spatial': pd.DataFrame(d, index=y.index).astype('float32')}


def horizon_x(base, h):
    target = base.index + pd.Timedelta(hours=h)
    d = {
        'hour_sin': np.sin(2 * np.pi * target.hour / 24),
        'hour_cos': np.cos(2 * np.pi * target.hour / 24),
        'year_sin': np.sin(2 * np.pi * target.day_of_year / 365.25),
        'year_cos': np.cos(2 * np.pi * target.day_of_year / 365.25),
        'month_sin': np.sin(2 * np.pi * (target.month - 1) / 12),
        'month_cos': np.cos(2 * np.pi * (target.month - 1) / 12),
        'weekday': target.day_of_week,
    }
    return np.column_stack([base.to_numpy(), *d.values()]).astype('float32'), list(base.columns) + list(d)


def metrics(pred, truth, threshold):
    assert pred.shape == truth.shape and np.isfinite(pred).all()
    error = pred - truth
    good = np.isfinite(truth)
    high = good & (truth >= threshold)
    called = good & (pred >= threshold)
    tp = (high & called).sum()
    prec = float(tp / called.sum()) if called.any() else 0.
    rec = float(tp / high.sum()) if high.any() else 0.
    return {'mae': float(np.nanmean(np.nanmean(np.abs(error), axis=0))),
            'late_mae': float(np.nanmean(np.nanmean(np.abs(error[:, 12:]), axis=0))),
            'rmse': float(np.sqrt(np.nanmean(error ** 2))),
            'bias': float(np.nanmean(error)), 'high_mae': float(np.mean(np.abs(error[high]))),
            'high_bias': float(np.mean(error[high])), 'precision': prec, 'recall': rec,
            'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.,
            'rows': int(good.sum()), 'high_rows': int(high.sum())}


def table(pred, index, model):
    H = 24
    issue = index + pd.Timedelta(hours=1)
    result = pd.DataFrame({'issue_time': np.repeat(issue, H), 'horizon_hours': np.tile(np.arange(1, H + 1), len(index)),
                           'y_pred': pred.reshape(-1), 'model_id': model})
    result['target_start'] = result.issue_time + pd.to_timedelta(result.horizon_hours - 1, unit='h')
    return result


def train_recipe(name, y, bases, trainpos, predictpos, stage, earlypos=None, rounds=None):
    group, objective, residual = RECIPES[name]
    out = RUN / 'models' / stage / name
    out.mkdir(parents=True, exist_ok=True)
    predictions, records = [], []
    params = CFG['params'] | {'objective': objective, 'metric': 'l1', 'num_threads': CFG['threads'], 'seed': CFG['seed']}
    for h in range(1, 25):
        path, meta = out / f'h{h:02d}.txt', out / f'h{h:02d}.json'
        x, columns = horizon_x(bases[group], h)
        target = y.shift(-h).to_numpy().copy()
        if residual:
            target -= y.to_numpy()
        tr = trainpos[np.isfinite(target[trainpos])]
        if path.exists():
            model = lgb.Booster(model_file=str(path))
            record = json.loads(meta.read_text(encoding='utf-8'))
        else:
            tick = time.perf_counter()
            trainset = lgb.Dataset(x[tr], label=target[tr], feature_name=columns)
            kwargs = {}
            if earlypos is not None:
                es = earlypos[np.isfinite(target[earlypos])]
                validset = lgb.Dataset(x[es], label=target[es], reference=trainset, feature_name=columns)
                kwargs = {'valid_sets': [validset], 'callbacks': [lgb.early_stopping(CFG['patience'], verbose=False)]}
            model = lgb.train(params, trainset, num_boost_round=CFG['max_rounds'] if rounds is None else rounds[h - 1], **kwargs)
            model.save_model(str(path))
            record = {'h': h, 'rounds': model.best_iteration or model.current_iteration(), 'rows': len(tr),
                      'fit_seconds': time.perf_counter() - tick, 'model_sha256': sha(path)}
            dump(meta, record)
        raw = model.predict(x[predictpos], num_threads=CFG['threads'])
        if residual:
            raw += y.to_numpy()[predictpos]
        predictions.append(np.maximum(raw, 0))
        records.append(record)
        print(f'{stage} {name} h={h:02d} rounds={record["rounds"]} seconds={record["fit_seconds"]:.2f}', flush=True)
    pred = np.column_stack(predictions)
    table(pred, y.index[predictpos], name).to_parquet(RUN / f'{stage}_{name}.parquet', index=False)
    dump(out / 'training.json', records)
    return pred, records


def development():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_snapshot()
    if (RUN / 'selection_freeze.json').exists():
        raise RuntimeError('Selection already frozen; do not rerun development in this run.')
    signature = {'code': code_hashes(), 'snapshot': sha(ROOT / 'snapshot_manifest.json')}
    sigpath = RUN / 'development_signature.json'
    if sigpath.exists():
        assert json.loads(sigpath.read_text(encoding='utf-8')) == signature, 'Changed code/data; choose a new run'
    else:
        dump(sigpath, signature)
    y, pm10, panel, stations = read_data(2024)
    bases = feature_frames(y, pm10, panel)
    dump(RUN / 'features.json', {k: list(v.columns) for k, v in bases.items()})
    trainpos = positions(y, CFG['training_start'], CFG['validation_start'])
    earlypos = positions(y, CFG['validation_start'], CFG['selection_start'])
    valpos = positions(y, CFG['validation_start'], CFG['test_start'])
    selectpos = positions(y, CFG['selection_start'], CFG['test_start'])
    selectmask = np.isin(valpos, selectpos)
    threshold = float(y.loc[y.index < pd.Timestamp(CFG['validation_start'], tz=TZ)].quantile(.9))
    truth = labels(y, valpos)
    candidates, iterations, rows = {}, {}, []
    for name in RECIPES:
        pred, rec = train_recipe(name, y, bases, trainpos, valpos, 'validation', earlypos=earlypos)
        candidates[name] = pred
        iterations[name] = [r['rounds'] for r in rec]
        for segment, mask in [('full_year', np.ones(len(valpos), dtype=bool)), ('selection_H2', selectmask)]:
            rows.append({'model': name, 'period': segment, **metrics(pred[mask], truth[mask], threshold)})
        pd.DataFrame(rows).to_csv(RUN / 'candidate_metrics.csv', index=False)
        print('CANDIDATE', name, rows[-1], flush=True)
    options = []
    for name, pred in candidates.items():
        options.append({'components': {name: 1.}, **metrics(pred[selectmask], truth[selectmask], threshold)})
    for a, b in itertools.combinations(candidates, 2):
        for w in [.25, .5, .75]:
            blend = w * candidates[a] + (1 - w) * candidates[b]
            options.append({'components': {a: w, b: 1 - w}, **metrics(blend[selectmask], truth[selectmask], threshold)})
    options.sort(key=lambda d: d['mae'])
    chosen = options[0]
    pred = sum(candidates[name] * weight for name, weight in chosen['components'].items())
    table(pred, y.index[valpos], 'challenger').to_parquet(RUN / 'validation_challenger.parquet', index=False)
    dump(RUN / 'selection_candidates.json', options)
    freeze = {'frozen_at': stamp(), 'selection': chosen, 'all_iterations': iterations, 'stations': stations,
              'config': CFG, 'code_hashes': code_hashes(), 'snapshot_sha256': sha(ROOT / 'snapshot_manifest.json'),
              'threshold_validation': threshold, 'training_issues': len(trainpos), 'early_stop_issues': len(earlypos),
              'selection_issues': len(selectpos), 'validation_issues': len(valpos),
              'test_policy': '2025 one-time public historical comparison; 2026 excluded and reserved',
              'full_validation_metrics': metrics(pred, truth, threshold)}
    dump(RUN / 'selection_freeze.json', freeze)
    print('FROZEN', json.dumps(chosen, ensure_ascii=False), flush=True)


def final_train():
    freeze = json.loads((RUN / 'selection_freeze.json').read_text(encoding='utf-8'))
    for name, digest in freeze['code_hashes'].items():
        assert sha(ROOT / name) == digest, f'Frozen source changed: {name}'
    verify_snapshot()
    if (RUN / 'final_predictions_complete.json').exists():
        raise RuntimeError('Final prediction already complete. Reuse existing result.')
    # Training and inference are executed together but never score test labels here.
    y, pm10, panel, _ = read_data(2025, freeze['stations'])
    bases = feature_frames(y, pm10, panel)
    trainpos = positions(y, CFG['training_start'], CFG['test_start'])
    testpos = positions(y, CFG['test_start'], CFG['test_end'])
    predictpos = testpos
    final = np.zeros((len(predictpos), 24))
    cost = []
    # Also refit local ablation, irrespective of challenge scores.
    names = list(dict.fromkeys([*freeze['selection']['components'], 'local_l1']))
    for name in names:
        pred, rec = train_recipe(name, y, bases, trainpos, predictpos, 'final', rounds=freeze['all_iterations'][name])
        if name in freeze['selection']['components']:
            final += freeze['selection']['components'][name] * pred
        cost.append({'model': name, 'fit_seconds': sum(r['fit_seconds'] for r in rec), 'training_issues': len(trainpos)})
    table(final, y.index[testpos], 'challenger').to_parquet(RUN / 'test_challenger.parquet', index=False)
    dump(RUN / 'final_predictions_complete.json', {'completed_at': stamp(), 'cost': cost, 'prediction_sha256': sha(RUN / 'test_challenger.parquet'),
                                                  'test_issues': len(testpos), 'train_issues': len(trainpos),
                                                  'reserved_from': CFG['reserved_from']})
    print('Final 2025 predictions complete; 2026 excluded entirely.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['snapshot', 'develop', 'final'])
    args = p.parse_args()
    {'snapshot': snapshot, 'develop': development, 'final': final_train}[args.command]()
