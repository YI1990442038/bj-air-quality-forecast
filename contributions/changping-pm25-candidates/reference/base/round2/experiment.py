"""Bounded second round. Observation IO is restricted to 2021..2024."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import pipeline as v1

RUN = HERE / 'run'
TZ = 'Asia/Shanghai'
START = '2021-01-24'
KINDS = ['dynamics_l1', 'dynamics_weighted', 'dynamics_q55']


def data(year, fit_year=None, stations=None):
    if not 2021 <= year <= 2024:
        raise ValueError('Only 2021..2024 observations are permitted in round2')
    parts, audit = [], []
    for yr in range(2021, year + 1):
        path = PARENT / f'data_snapshot/observations_{yr}.parquet'
        f = pd.read_parquet(path, columns=['hour_start', 'source_station_name', 'station_version', 'pm25', 'pm10'])
        parts.append(f[f.station_version.eq('new_network')])
        audit.append({'path': str(path), 'sha256': v1.sha(path)})
    f = pd.concat(parts, ignore_index=True)
    assert not f.duplicated(['hour_start', 'source_station_name']).any()
    axis = pd.date_range('2017-01-01', f'{year+1}-01-01', freq='h', inclusive='left', tz=TZ)
    target = f[f.source_station_name.eq('昌平镇')].set_index('hour_start').reindex(axis)
    panel = f.pivot(index='hour_start', columns='source_station_name', values='pm25').reindex(axis)
    if stations is None:
        cut = pd.Timestamp(f'{(fit_year or year-1)+1}-01-01', tz=TZ)
        fit = panel.loc[(panel.index >= pd.Timestamp(START, tz=TZ)) & (panel.index < cut)]
        coverage = fit.notna().mean()
        stations = sorted(coverage[coverage.ge(.8)].index.tolist())
    panel = panel.reindex(columns=stations).astype('float32')
    return target.pm25.astype('float32'), target.pm10.astype('float32'), panel, audit


def dynamic_features(y, pm10, panel):
    base = v1.feature_frames(y, pm10, panel)['spatial']
    d = {}
    for window in [3, 6, 12, 24, 72]:
        avg = y.rolling(window, min_periods=1).mean()
        std = y.rolling(window, min_periods=1).std(ddof=0)
        d[f'dyn_deviation_{window}'] = y - avg
        d[f'dyn_standardized_{window}'] = (y - avg) / (std + 5)
    for lag in [1, 3, 6, 12]:
        velocity = y - y.shift(lag)
        d[f'dyn_acceleration_{lag}'] = velocity - velocity.shift(lag)
        d[f'dyn_relative_velocity_{lag}'] = velocity / (y.shift(lag) + 10)
    peers = panel.drop(columns=['昌平镇'], errors='ignore')
    gap = peers.sub(y, axis=0)
    for q in [.1, .25, .75, .9]:
        d[f'dyn_peer_gap_q{int(q*100)}'] = gap.quantile(q, axis=1)
    valid_count = peers.notna().sum(axis=1).replace(0, np.nan)
    d['dyn_peer_above_local_10'] = gap.gt(10).sum(axis=1) / valid_count
    d['dyn_peer_below_local_10'] = gap.lt(-10).sum(axis=1) / valid_count
    for lag in [1, 3, 6, 12]:
        change = peers - peers.shift(lag)
        valid_change = change.notna().sum(axis=1).replace(0, np.nan)
        d[f'dyn_peer_rise_fraction_{lag}'] = change.gt(5).sum(axis=1) / valid_change
        d[f'dyn_peer_fall_fraction_{lag}'] = change.lt(-5).sum(axis=1) / valid_change
        d[f'dyn_peer_change_mean_{lag}'] = change.mean(axis=1)
        d[f'dyn_peer_change_q90_{lag}'] = change.quantile(.9, axis=1)
        d[f'dyn_peer_change_minus_local_{lag}'] = change.mean(axis=1) - (y - y.shift(lag))
    return pd.concat([base, pd.DataFrame(d, index=y.index).astype('float32')], axis=1)


def fit_predict(kind, y, base, trainpos, earlypos, predpos, threshold, folder, fixed_rounds=None):
    folder.mkdir(parents=True, exist_ok=True)
    residual = kind != 'base_absolute'
    objective = 'quantile' if kind == 'dynamics_q55' else 'regression_l1'
    params = v1.CFG['params'] | {'objective': objective, 'metric': 'l1', 'num_threads': 2, 'seed': 20261026}
    if objective == 'quantile':
        params['alpha'] = .55
    predictions, rounds = [], []
    for h in range(1, 25):
        x, cols = v1.horizon_x(base, h)
        absolute = y.shift(-h).to_numpy()
        target = absolute - y.to_numpy() if residual else absolute
        tr = trainpos[np.isfinite(target[trainpos])]
        path, meta = folder / f'h{h:02d}.txt', folder / f'h{h:02d}.json'
        if path.exists():
            info = json.loads(meta.read_text(encoding='utf-8'))
            assert v1.sha(path) == info['sha256']
            model = lgb.Booster(model_file=str(path))
        else:
            tick = time.perf_counter()
            weights = 1 + .25 * np.minimum(absolute[tr] / threshold, 3) if kind == 'dynamics_weighted' else None
            trainset = lgb.Dataset(x[tr], label=target[tr], weight=weights, feature_name=cols)
            kwargs = {}
            if fixed_rounds is None:
                es = earlypos[np.isfinite(target[earlypos])]
                valset = lgb.Dataset(x[es], label=target[es], reference=trainset, feature_name=cols)
                kwargs = {'valid_sets': [valset], 'callbacks': [lgb.early_stopping(50, verbose=False)]}
            model = lgb.train(params, trainset, num_boost_round=500 if fixed_rounds is None else fixed_rounds[h-1], **kwargs)
            model.save_model(str(path))
            info = {'h': h, 'rounds': model.best_iteration or model.current_iteration(), 'train_rows': len(tr),
                    'fit_seconds': time.perf_counter() - tick, 'sha256': v1.sha(path)}
            v1.dump(meta, info)
        p = model.predict(x[predpos], num_threads=2)
        if residual:
            p += y.to_numpy()[predpos]
        predictions.append(np.maximum(p, 0))
        rounds.append(info['rounds'])
        print(f'{folder.parent.name}/{kind} h={h:02d} rounds={info["rounds"]} sec={info["fit_seconds"]:.2f}', flush=True)
    return np.column_stack(predictions), rounds


def load_v1_prediction(name, y, pos):
    path = PARENT / f'runs/challenge_v1/validation_{name}.parquet'
    df = pd.read_parquet(path).sort_values(['issue_time', 'horizon_hours'])
    assert df.issue_time.dt.year.eq(2024).all()
    assert len(df) == 24 * len(pos)
    assert pd.DatetimeIndex(df.issue_time[::24]).equals(y.index[pos] + pd.Timedelta(hours=1))
    return df.y_pred.to_numpy().reshape(-1, 24)


def diagnose(predictions, y, pos, threshold, year):
    truth = v1.labels(y, pos)
    now = y.to_numpy()[pos, None]
    rows = []
    conditions = {
        'already_high': np.broadcast_to(now >= threshold, truth.shape),
        'new_high': (now < threshold) & (truth >= threshold),
        'large_rise_ge30': truth - now >= 30,
        'large_fall_ge30': truth - now <= -30,
        'other': np.abs(truth - now) < 30,
    }
    for name, pred in predictions.items():
        err = pred - truth
        for condition, mask in conditions.items():
            mask = mask & np.isfinite(truth)
            rows.append({'year': year, 'model': name, 'condition': condition, 'rows': int(mask.sum()),
                         'mae': float(np.mean(np.abs(err[mask]))), 'bias': float(np.mean(err[mask]))})
    return rows


def develop():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'selection.json').exists():
        raise RuntimeError('Round2 already selected; preserve it.')
    signature = {'experiment.py': v1.sha(Path(__file__)), 'PROTOCOL.md': v1.sha(HERE / 'PROTOCOL.md'),
                 'v1_pipeline': v1.sha(PARENT / 'pipeline.py'), 'v1_config': v1.sha(PARENT / 'config.json'),
                 'data': {str(y): v1.sha(PARENT / f'data_snapshot/observations_{y}.parquet') for y in range(2021, 2025)}}
    sigpath = RUN / 'signature.json'
    if sigpath.exists():
        assert json.loads(sigpath.read_text()) == signature
    else:
        v1.dump(sigpath, signature)
    folds, all_rows, diagnostics, final_rounds = {}, [], [], {}
    for year in [2023, 2024]:
        y, pm10, panel, audit = data(year)
        left, half, right = f'{year}-01-01', f'{year}-07-01', f'{year+1}-01-01'
        trainpos = v1.positions(y, START, left)
        earlypos = v1.positions(y, left, half)
        pos = v1.positions(y, left, right)
        selectpos = v1.positions(y, half, right)
        mask = np.isin(pos, selectpos)
        threshold = float(y[y.index < pd.Timestamp(left, tz=TZ)].quantile(.9))
        truth = v1.labels(y, pos)
        base = v1.feature_frames(y, pm10, panel)['spatial']
        rich = dynamic_features(y, pm10, panel)
        v1.dump(RUN / f'fold_{year}.json', {'year': year, 'stations': list(panel.columns), 'data_inputs': audit,
                 'train_issues': len(trainpos), 'early_stop_issues': len(earlypos), 'selection_issues': len(selectpos),
                 'threshold_train_p90': threshold, 'rich_feature_columns': list(rich.columns), 'base_feature_columns': list(base.columns)})
        prediction = {}
        if year == 2024:
            absolute = load_v1_prediction('spatial_l1', y, pos)
            delta = load_v1_prediction('spatial_delta_l1', y, pos)
        else:
            absolute, _ = fit_predict('base_absolute', y, base, trainpos, earlypos, pos, threshold, RUN / 'models' / str(year) / 'base_absolute')
            delta, _ = fit_predict('base_delta', y, base, trainpos, earlypos, pos, threshold, RUN / 'models' / str(year) / 'base_delta')
        prediction['v1'] = .25 * absolute + .75 * delta
        prediction['delta_only'] = delta
        prediction['persistence'] = np.broadcast_to(y.to_numpy()[pos, None], truth.shape).copy()
        for kind in KINDS:
            pred, rounds = fit_predict(kind, y, rich, trainpos, earlypos, pos, threshold, RUN / 'models' / str(year) / kind)
            prediction[kind] = pred
            if year == 2024:
                final_rounds[kind] = rounds
            for period, m in [('full_year', np.ones(len(pos), dtype=bool)), ('H2_selection', mask)]:
                row = {'year': year, 'model': kind, 'period': period, **v1.metrics(pred[m], truth[m], threshold)}
                all_rows.append(row)
            print('CANDIDATE', all_rows[-1], flush=True)
            pd.DataFrame(all_rows).to_csv(RUN / 'base_metrics.csv', index=False)
        for name in ['v1', 'delta_only', 'persistence']:
            for period, m in [('full_year', np.ones(len(pos), dtype=bool)), ('H2_selection', mask)]:
                all_rows.append({'year': year, 'model': name, 'period': period, **v1.metrics(prediction[name][m], truth[m], threshold)})
        for name, pred in prediction.items():
            v1.table(pred, y.index[pos], name).to_parquet(RUN / f'prediction_{year}_{name}.parquet', index=False)
        diagnostics += diagnose(prediction, y, pos, threshold, year)
        folds[year] = {'y': y, 'pos': pos, 'mask': mask, 'threshold': threshold, 'truth': truth, 'predictions': prediction}
    pd.DataFrame(all_rows).to_csv(RUN / 'base_metrics.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(RUN / 'error_diagnosis.csv', index=False)
    options = [{'candidate': name, 'weight': w, 'gate': False} for name in ['delta_only', *KINDS] for w in [.25, .5, .75, 1.]]
    options += [{'candidate': 'persistence', 'weight': w, 'gate': True} for w in [.2, .4]]
    rows = []
    for opt in options:
        score = opt.copy()
        ratios, passed = [], True
        for year, f in folds.items():
            base, cand = f['predictions']['v1'], f['predictions'][opt['candidate']]
            weight = opt['weight']
            if opt['gate']:
                weight = weight * (f['y'].to_numpy()[f['pos'], None] >= f['threshold'])
            pred = (1 - weight) * base + weight * cand
            m = f['mask']
            old = v1.metrics(base[m], f['truth'][m], f['threshold'])
            new = v1.metrics(pred[m], f['truth'][m], f['threshold'])
            passed &= new['mae'] <= old['mae'] * 1.01 and new['late_mae'] <= old['late_mae'] * 1.01 and new['high_mae'] <= old['high_mae'] * .95
            ratios.append(new['mae'] / old['mae'])
            score[str(year)] = new | {'mae_ratio': ratios[-1], 'high_mae_ratio': new['high_mae'] / old['high_mae'], 'late_mae_ratio': new['late_mae'] / old['late_mae']}
        score['passes'] = bool(passed)
        score['mean_mae_ratio'] = float(np.mean(ratios))
        rows.append(score)
    rows.sort(key=lambda d: (not d['passes'], d['mean_mae_ratio']))
    chosen = rows[0] if rows[0]['passes'] else None
    v1.dump(RUN / 'all_candidates.json', rows)
    selection = {'selected_at': v1.stamp(), 'selected': chosen, 'rounds_2024': final_rounds, 'signature': signature,
                 'status': 'development_candidate_pending_reserved_evaluation' if chosen else 'retain_v1_no_candidate_passed',
                 '2025_opened': False, '2026_opened': False}
    v1.dump(RUN / 'selection.json', selection)
    if chosen:
        for year, f in folds.items():
            weight = chosen['weight']
            if chosen['gate']:
                weight *= (f['y'].to_numpy()[f['pos'], None] >= f['threshold'])
            pred = (1 - weight) * f['predictions']['v1'] + weight * f['predictions'][chosen['candidate']]
            v1.table(pred, f['y'].index[f['pos']], 'v2_candidate').to_parquet(RUN / f'prediction_{year}_selected.parquet', index=False)
    print('SELECTION', json.dumps(selection['selected'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['develop'])
    args = parser.parse_args()
    develop()
