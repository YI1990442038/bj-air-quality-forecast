"""Fit two high-value alert models and score their frozen operating points."""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'c25_alert'))

import train_alert as alert
from local_run import ADDED

base = alert.parent
np, pd, lgb, Q = alert.np, alert.pd, alert.lgb, alert.Q
RUN = HERE / 'run'
FEATURES = base.FEATURES + ['c25_concentration', 'c25_uplift']
PARAMS = dict(
    objective='binary', metric='None', learning_rate=.05, max_depth=3, num_leaves=7,
    min_data_in_leaf=400, lambda_l2=10, max_bin=63, seed=20261026,
    num_threads=2, deterministic=True, force_col_wise=True, verbosity=-1,
)


def matrix(frozen, round1, r140, c25, bounds):
    issues, x, truth, values = base.baseline_grid(frozen, round1, r140, bounds)
    corrected = base.apply(c25, x, values['R1-40'])
    uplift = corrected - values['R1-40']
    issue = issues.repeat(24)
    horizon = np.tile(base.HORIZONS, len(issues))
    hours = issue + pd.to_timedelta(horizon - 1, unit='h')
    x = np.column_stack([x, corrected.ravel(), uplift.ravel()])
    return x, truth.ravel(), hours, issue, horizon, corrected.ravel()


def fit(kind, x, high, hours):
    if kind == 'logistic':
        return alert.fit(x, high, hours)
    counts = pd.Series(hours).value_counts()
    weights = 1. / counts.reindex(hours).to_numpy()
    return lgb.train(PARAMS, lgb.Dataset(x, label=high, weight=weights, feature_name=FEATURES),
                     num_boost_round=80)


def predict(kind, model, x):
    if kind == 'logistic':
        return alert.probability(model, x)
    if model.feature_name() != FEATURES:
        raise ValueError('Unexpected alert feature order')
    return model.predict(x, num_threads=2)


def fit_fixed_alerts(grids):
    x = np.concatenate([g[0] for g in grids])
    truth = np.concatenate([g[1] for g in grids])
    hours = grids[0][2].append(grids[1][2])
    labelled = np.isfinite(truth)
    x, high, hours = x[labelled], truth[labelled] >= Q, hours[labelled]
    month = hours.month.to_numpy()
    summary = {}
    for kind in ('logistic', 'tree'):
        tick = time.monotonic()
        oof = np.full(len(high), np.nan)
        for held_month in (4, 5, 6):
            held = month == held_month
            model = fit(kind, x[~held], high[~held], hours[~held])
            oof[held] = predict(kind, model, x[held])
        high_scores = np.sort(oof[(month >= 4) & high])[::-1]
        if not len(high_scores) or not np.isfinite(high_scores).all():
            raise ValueError('Incomplete out-of-fold scores on high labels')
        thresholds = {
            f'{kind}-{pct}': float(high_scores[int(np.ceil(pct / 100 * len(high_scores))) - 1])
            for pct in (70, 80)
        }
        model = fit(kind, x, high, hours)
        path = RUN / f'{kind}.txt'
        if kind == 'logistic':
            path = RUN / 'logistic.json'
            base.write_new(path, model)
        else:
            model.save_model(str(path))
        summary[kind] = {
            'thresholds': thresholds, 'weights': path.name, 'sha256': base.sha(path),
            'fit_rows': len(high), 'high_fit_rows': int(high.sum()),
            'calibration_high_rows': len(high_scores), 'seconds': time.monotonic() - tick,
        }
    return summary


def alert_metrics(alarm, high):
    tp = int((alarm & high).sum())
    fp = int((alarm & ~high).sum())
    return {
        'precision': tp / max(1, tp + fp), 'recall': tp / int(high.sum()),
        'false_positive_rate': fp / int((~high).sum()), 'tp': tp, 'fp': fp,
    }


def at_reference_recall(score, high, target_tp):
    if target_tp < 1:
        raise ValueError('Positive reference recall is required')
    cutoff = np.sort(score[high])[-target_tp]
    return alert_metrics(score >= cutoff, high)


def replay(grid, models):
    x, truth, hours, issue, horizon, corrected = grid
    known = np.isfinite(truth)
    high = truth[known] >= Q
    baseline = alert_metrics(corrected[known] >= Q, high)
    metrics = {'C25': baseline}
    diagnostic = {'C25': at_reference_recall(corrected[known], high, baseline['tp'])}
    data = {
        'issue_time': issue, 'horizon_hours': horizon, 'truth': truth,
        'C25_concentration': corrected, 'C25_alarm': corrected >= Q,
    }
    for kind, entry in models.items():
        path = RUN / entry['weights']
        if base.sha(path) != entry['sha256']:
            raise ValueError('Frozen alert model hash mismatch')
        model = base.read(path) if kind == 'logistic' else lgb.Booster(model_file=str(path))
        score = predict(kind, model, x)
        if not np.isfinite(score).all():
            raise ValueError('Missing alert scores')
        data[f'{kind}_probability'] = score
        diagnostic[kind] = at_reference_recall(score[known], high, baseline['tp'])
        for name, threshold in entry['thresholds'].items():
            alarm = score >= threshold
            data[name] = alarm
            metrics[name] = alert_metrics(alarm[known], high)
    passed = {
        name: m['precision'] > baseline['precision'] and m['recall'] >= .70
        and m['false_positive_rate'] < baseline['false_positive_rate']
        for name, m in metrics.items() if name != 'C25'
    }
    eligible = [name for name, ok in passed.items() if ok]
    selected = max(eligible, key=lambda n: (metrics[n]['precision'], metrics[n]['recall'])) if eligible else None
    pd.DataFrame(data).to_parquet(RUN / 'replay_predictions.parquet', index=False)
    base.write_new(RUN / 'replay.json', {
        'bounds': base.HOLDOUT, 'status': 'reused_development',
        'scored_rows': int(known.sum()), 'high_target_hours': hours[known][high].nunique(),
        'metrics': metrics, 'passes': passed, 'selected': selected,
        'at_c25_recall_diagnostic': diagnostic, 'fits_sha256': base.sha(RUN / 'fits.json'),
    })
    print('Selected', selected, flush=True)


def main():
    RUN.mkdir(exist_ok=False)
    prior = base.read(alert.RESIDUAL / 'run/execution.json')
    pins = base.read(alert.RESIDUAL / 'run/replay.json')['model_sha256']
    r140_path = HERE.parent / 'round1_refinement/run/R1-40.txt'
    c25_path = alert.RESIDUAL / 'run/R140-C25.txt'
    round1, round1_pin = base.parent_model()
    if (round1_pin != prior['round1_sha256'] or base.sha(r140_path) != prior['r140_sha256']
            or base.sha(c25_path) != pins['R140-C25']):
        raise ValueError('Frozen parent hash mismatch')
    base.write_new(RUN / 'execution.json', {
        'protocol_sha256': base.sha(HERE / 'PROTOCOL.md'), 'source_sha256': base.sha(__file__),
        'c25_sha256': pins['R140-C25'], 'r140_sha256': prior['r140_sha256'],
        'round1_sha256': round1_pin, 'maximum_fits': 8,
    })
    frozen = base.FrozenM1()
    r140 = lgb.Booster(model_file=str(r140_path))
    c25 = lgb.Booster(model_file=str(c25_path))
    grids = [matrix(frozen, round1, r140, c25, bounds) for bounds in (ADDED, base.VALIDATION)]
    models = fit_fixed_alerts(grids)
    base.write_new(RUN / 'fits.json', models)
    print(models, flush=True)
    replay(matrix(frozen, round1, r140, c25, base.HOLDOUT), base.read(RUN / 'fits.json'))


if __name__ == '__main__':
    main()
