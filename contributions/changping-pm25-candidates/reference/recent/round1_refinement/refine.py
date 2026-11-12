"""Fit two small blend models on Q1 and apply the frozen evaluation protocol."""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from local_run import (
    ADDED, FEATURES, HOLDOUT, HORIZONS, THRESHOLD, VALIDATION, FrozenM1,
    diagnostic, lgb, np, passes, pd, read, sha, timestamp, write_new,
)

PARENT_RUN = HERE.parent / 'run'
RUN = HERE / 'run'
SHARES = {'R1-25': .25, 'R1-40': .40}
GATE_FEATURES = FEATURES + ['m1', 'r1_correction']
PARAMS = dict(
    objective='regression_l1', metric='None', device_type='cpu',
    learning_rate=.05, max_depth=3, num_leaves=7, min_data_in_leaf=500,
    lambda_l2=10, max_bin=63, seed=20261026, num_threads=2,
    deterministic=True, force_col_wise=True, verbosity=-1,
)


def parent_model():
    path = PARENT_RUN / 'round1.txt'
    pin = read(PARENT_RUN / 'validation.json')['model_sha256']['1']
    if sha(path) != pin:
        raise ValueError('The original Round 1 weight hash changed')
    model = lgb.Booster(model_file=str(path))
    if model.feature_name() != FEATURES:
        raise ValueError('The original Round 1 feature order changed')
    return model, pin


def make_grid(frozen, parent, bounds):
    issues, x, truth, values, _ = frozen.grid(bounds, [bounds])
    base = values['M1'].ravel()
    round1 = np.maximum(0., base + np.clip(parent.predict(x, num_threads=2), -15., 15.))
    if not np.isfinite(round1).all():
        raise ValueError('Round 1 did not produce a finite forecast on the shared grid')
    values['Round1'] = round1.reshape(truth.shape)
    change = round1 - base
    return issues, np.column_stack([x, base, change]).astype('float32'), truth, values


def training_data(grid, share):
    issues, x, truth, values = grid
    base = values['M1'].ravel()
    change = values['Round1'].ravel() - base
    actual = truth.ravel()
    good = np.isfinite(actual) & (np.abs(change) > 1e-9)
    x, actual, base, change = x[good], actual[good], base[good], change[good]
    target = np.clip((actual - base) / change, 0., 1.)
    ends = issues.repeat(24) + pd.to_timedelta(np.tile(HORIZONS, len(issues)), unit='h')
    age = (timestamp(ADDED[1]) - ends[good]) / pd.Timedelta(days=1)
    recency = np.exp2(-np.asarray(age) / 180.)
    high = actual >= THRESHOLD
    if not high.any() or high.all():
        raise ValueError('Both label groups are required')
    weight = np.where(
        high, share * recency / recency[high].sum(),
        (1 - share) * recency / recency[~high].sum(),
    ) * np.abs(change)
    weight /= weight.mean()
    return x, target, weight, int(high.sum())


def fit_candidates(grid, parent_pin):
    pins = {}
    for name, share in SHARES.items():
        x, target, weight, high_rows = training_data(grid, share)
        write_new(RUN / f'{name}_started.json', {
            'parent_sha256': parent_pin, 'fit_bounds': ADDED, 'rows': len(target),
            'high_rows': high_rows, 'high_loss_share': share, 'params': PARAMS,
        })
        tick = time.monotonic()
        model = lgb.train(
            PARAMS, lgb.Dataset(x, label=target, weight=weight, feature_name=GATE_FEATURES),
            num_boost_round=80,
        )
        seconds = time.monotonic() - tick
        path = RUN / f'{name}.txt'
        model.save_model(str(path))
        pins[name] = sha(path)
        write_new(RUN / f'{name}_fit.json', {
            'seconds': seconds, 'trees': model.num_trees(), 'model_sha256': pins[name],
        })
        print(f'{name} fitted {len(target):,} rows and {model.num_trees()} trees in {seconds:.3f}s', flush=True)
    return pins


def score(grid, pins):
    issues, x, truth, values = grid
    for name, pin in pins.items():
        path = RUN / f'{name}.txt'
        if sha(path) != pin:
            raise ValueError('The selected weight hash changed')
        model = lgb.Booster(model_file=str(path))
        if model.feature_name() != GATE_FEATURES:
            raise ValueError('The refinement feature order changed')
        alpha = np.clip(model.predict(x, num_threads=2).reshape(truth.shape), 0., 1.)
        values[name] = values['M1'] + alpha * (values['Round1'] - values['M1'])
        if not np.isfinite(values[name]).all():
            raise ValueError('The refinement produced an incomplete forecast grid')
    current = values['persistence'][:, 0]
    metrics = {}
    for name, prediction in values.items():
        result = diagnostic(prediction, truth, THRESHOLD, current)
        result['late_mae'] = float(np.nanmean(np.nanmean(np.abs(prediction[:, 12:] - truth[:, 12:]), axis=0)))
        metrics[name] = result
    issue = issues.repeat(24)
    horizons = np.tile(HORIZONS, len(issues))
    target_hours = issue + pd.to_timedelta(horizons - 1, unit='h')
    high_hours = target_hours[truth.ravel() >= THRESHOLD].unique()
    high_days = high_hours.normalize().nunique()
    support = len(high_hours) >= 24 and high_days >= 5 and 'ordinary_mae' in metrics['M1']
    gates = {name: passes(metrics[name], metrics['M1'], support) for name in pins}
    eligible = [name for name in pins if gates[name]]
    selected = min(eligible, key=lambda n: (metrics[n]['high_mae'], metrics[n]['mae'], SHARES[n])) if eligible else None
    forecasts = pd.DataFrame({
        'issue_time': issue, 'horizon_hours': horizons, 'truth': truth.ravel(),
    } | {name: prediction.ravel() for name, prediction in values.items()})
    return {
        'issues': len(issues), 'high_target_hours': len(high_hours), 'high_target_days': high_days,
        'model_sha256': pins, 'metrics': metrics, 'passes': gates, 'selected': selected,
    }, forecasts


def evaluate(frozen, parent, bounds, pins, stage):
    result, forecasts = score(make_grid(frozen, parent, bounds), pins)
    forecasts.to_parquet(RUN / f'{stage}_predictions.parquet', index=False)
    result['bounds'] = bounds
    write_new(RUN / f'{stage}.json', result)
    print(stage, result['passes'], 'selected=', result['selected'], flush=True)
    return result


def main():
    RUN.mkdir(exist_ok=False)
    frozen = FrozenM1()
    parent, parent_pin = parent_model()
    write_new(RUN / 'execution.json', {
        'parent_sha256': parent_pin, 'source_sha256': sha(__file__),
        'protocol_sha256': sha(HERE / 'PROTOCOL.md'), 'lightgbm': lgb.__version__,
    })
    pins = fit_candidates(make_grid(frozen, parent, ADDED), parent_pin)
    validation = evaluate(frozen, parent, VALIDATION, pins, 'validation')
    selected = validation['selected']
    if selected is None:
        return
    write_new(RUN / 'holdout_started.json', {
        'candidate': selected, 'bounds': HOLDOUT, 'validation_sha256': sha(RUN / 'validation.json'),
    })
    evaluate(frozen, parent, HOLDOUT, {selected: pins[selected]}, 'holdout')


if __name__ == '__main__':
    main()
