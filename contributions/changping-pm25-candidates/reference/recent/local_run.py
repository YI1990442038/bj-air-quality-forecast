"""Prepare train-only matrices and score the fixed M1 candidates locally."""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CHALLENGER = PROJECT / 'experiments/changping_pm25_challenger'
sys.path.insert(0, str(CHALLENGER))

from v2_moe_astra_20261030.contracts import TZ, blend, checked_models, read, sha, verify_fold, write_new
from v2_focus_20261029.driver import pure_features
from m1_high_priority_20261030.dependencies import load_reused
from m1_high_priority_20261030.driver import selected_predict
from m1_continuous_loss_CL1_20261031.evaluation import diagnostic

import lightgbm as lgb
import numpy as np
import pandas as pd

YEAR = 2024
THRESHOLD = 73
HOUR = pd.Timedelta(hours=1)
HORIZONS = np.arange(1, 25)
TRAIN = ('2025-01-01', '2026-01-01')
ADDED = ('2026-01-01', '2026-04-01')
VALIDATION = ('2026-04-01', '2026-07-01')
HOLDOUT = ('2026-07-01', '2026-10-26')
RUN = HERE / 'run'
FEATURES = [
    'v2', 'high_expert', 'old_probability', 'expert_gap', 'horizon_hours',
    'target_hour_sin', 'target_hour_cos', 'local_latest', 'local_change3',
    'local_gap_from_peak6', 'peer_median', 'peer_high_fraction',
    'peer_high_fraction_change3', 'peer_valid_count',
]


def timestamp(date):
    return pd.Timestamp(date, tz=TZ)


def read_window(bounds, stations, asset):
    left, right = map(timestamp, bounds)
    start = left - 512 * HOUR
    columns = ['hour_start', 'source_station_name', 'pm25', 'pm10']
    pieces = []
    for year in range(start.year, (right - HOUR).year + 1):
        name = f'observations_{year}.parquet'
        if year == YEAR:
            path = CHALLENGER / 'data_snapshot' / name
            if sha(path) != asset['files'][f'data_snapshot/{name}']:
                raise ValueError('Frozen 2024 observation hash mismatch')
        else:
            path = PROJECT / 'data/air_quality/beijing/processed' / name
        pieces.append(pd.read_parquet(path, columns=columns, engine='pyarrow', filters=[
            ('hour_start', '>=', start), ('hour_start', '<', right),
            ('station_version', '==', 'new_network'),
            ('source_station_name', 'in', sorted(set(stations) | {'昌平镇'})),
        ]))
    observed = pd.concat(pieces, ignore_index=True)
    if (not observed.hour_start.between(start, right, inclusive='left').all()
            or observed.duplicated(['hour_start', 'source_station_name']).any()):
        raise ValueError('Observation date or station/hour key mismatch')
    axis = pd.date_range(start, right, freq='h', inclusive='left')
    target = observed.loc[observed.source_station_name == '昌平镇'].set_index('hour_start').reindex(axis)
    panel = observed.pivot(index='hour_start', columns='source_station_name', values='pm25')
    return target.pm25.astype('float32'), target.pm10.astype('float32'), panel.reindex(
        index=axis, columns=stations).astype('float32')


def issue_positions(y, bounds):
    left, right = map(timestamp, bounds)
    issue = y.index + HOUR
    recent_complete = y.rolling(24, min_periods=24).count().eq(24).to_numpy()
    return np.flatnonzero(
        recent_complete & (np.arange(len(y)) >= 511) & (issue >= left)
        & (issue + 24 * HOUR <= right) & (np.arange(len(y)) + 24 < len(y))
    )


def correction_features(y, panel, positions, v2, high, probability):
    peers = panel.drop(columns='昌平镇', errors='ignore')
    count = peers.notna().sum(axis=1)
    fraction = peers.ge(THRESHOLD).sum(axis=1) / count.replace(0, np.nan)
    recent = pd.DataFrame({
        'local_latest': y,
        'local_change3': y - y.shift(3),
        'local_gap_from_peak6': y.rolling(6, min_periods=1).max() - y,
        'peer_median': peers.median(axis=1),
        'peer_high_fraction': fraction,
        'peer_high_fraction_change3': fraction - fraction.shift(3),
        'peer_valid_count': count,
    }).iloc[positions]
    hour = (y.index[positions].hour.to_numpy()[:, None] + HORIZONS) % 24
    angle = 2 * np.pi * hour / 24
    x = np.column_stack([
        v2.ravel(), high.ravel(), probability.ravel(), (high - v2).ravel(),
        np.tile(HORIZONS, len(positions)), np.sin(angle).ravel(), np.cos(angle).ravel(),
        np.repeat(recent.to_numpy(), 24, axis=0),
    ]).astype('float32')
    if x.shape != (len(positions) * 24, len(FEATURES)) or np.isinf(x).any():
        raise ValueError('Invalid correction feature matrix')
    return x


class FrozenM1:
    def __init__(self):
        self.asset = read(CHALLENGER / 'm1_high_priority_20261030/INPUT_MANIFEST.json')
        self.fold = verify_fold(YEAR, self.asset)
        if self.fold['threshold_train_p90'] != THRESHOLD:
            raise ValueError('Frozen threshold mismatch')
        self.stations = self.fold['stations']
        self.models = checked_models(
            YEAR, self.asset, self.stations, self.fold['base_feature_columns'],
            self.fold['rich_feature_columns'],
        )
        self.highs, self.gates = load_reused(YEAR, self.fold, self.asset)

    def grid(self, read_bounds, periods):
        y, pm10, panel = read_window(read_bounds, self.stations, self.asset)
        groups = [issue_positions(y, bounds) for bounds in periods]
        pos = np.concatenate(groups)
        if not len(pos) or np.any(np.diff(pos) <= 0):
            raise ValueError('Missing or overlapping issue times')
        plain, rich, core = pure_features(y, pm10, panel)
        if (list(plain.columns) != self.fold['base_feature_columns']
                or list(rich.columns) != self.fold['rich_feature_columns']):
            raise ValueError('Feature order differs from the frozen fold')
        v2, high, probability = selected_predict(self.models, self.highs, self.gates, plain, rich, y, pos, core)
        x = correction_features(y, panel, pos, v2, high, probability)
        current = y.to_numpy()[pos]
        truth = y.to_numpy()[pos[:, None] + HORIZONS]
        values = {
            'M1': blend(probability, v2, high),
            'V2': v2,
            'persistence': np.broadcast_to(current[:, None], truth.shape),
        }
        return y.index[pos] + HOUR, x, truth, values, [len(group) for group in groups]

    def verify_replay(self):
        period = ('2024-12-01', '2024-12-14')
        issues, _, _, base, _ = self.grid(period, [period])
        path = CHALLENGER / 'v2_moe_astra_20261030/runs/moe_M1/fold_2024/predictions.parquet'
        expected = pd.read_parquet(path, columns=['issue_time', 'horizon_hours', 'y_pred'], filters=[
            ('issue_time', '>=', issues[0]), ('issue_time', '<=', issues[-1]),
        ])
        keys = pd.MultiIndex.from_product([issues, HORIZONS], names=['issue_time', 'horizon_hours'])
        actual_keys = pd.MultiIndex.from_frame(expected[['issue_time', 'horizon_hours']])
        if not actual_keys.equals(keys):
            raise ValueError('Frozen replay does not reproduce the original issue grid')
        difference = float(np.max(np.abs(base['M1'].ravel() - expected.y_pred.to_numpy())))
        if difference > 1e-6:
            raise ValueError(f'Frozen replay prediction difference {difference}')
        return {'trajectories': len(issues), 'max_m1_difference': difference}


def train_weights(truth, target_times, end):
    age_days = (timestamp(end) - target_times) / pd.Timedelta(days=1)
    recency = np.exp2(-np.asarray(age_days) / 180)
    high = truth >= THRESHOLD
    weights = np.empty(len(truth), dtype='float64')
    for mask, share in ((high, .60), (~high, .40)):
        if not mask.any():
            raise ValueError('Training requires both high and ordinary observations')
        weights[mask] = share * len(truth) * recency[mask] / recency[mask].sum()
    return weights.astype('float32')


def prepare():
    started = time.monotonic()
    frozen = FrozenM1()
    replay = frozen.verify_replay()
    issues, x, truth, values, groups = frozen.grid((TRAIN[0], ADDED[1]), [TRAIN, ADDED])
    target_ends = issues.repeat(24) + pd.to_timedelta(np.tile(HORIZONS, len(issues)), unit='h')
    observed = np.isfinite(truth.ravel())
    n1 = int(observed[:groups[0] * 24].sum())
    x, y, residual = x[observed], truth.ravel()[observed], (truth - values['M1']).ravel()[observed]
    dates = target_ends[observed]
    w1 = train_weights(y[:n1], dates[:n1], TRAIN[1])
    w2 = train_weights(y, dates, ADDED[1])
    RUN.mkdir(exist_ok=False)
    path = RUN / 'm1_fit_inputs.npz'
    np.savez_compressed(path, x=x, residual=residual.astype('float32'), w1=w1, w2=w2, features=FEATURES)
    write_new(RUN / 'preparation.json', {
        'replay': replay, 'training_bounds': [TRAIN, ADDED], 'matrix_sha256': sha(path),
        'fit_rows': [n1, len(y)], 'high_rows': [int((y[:n1] >= THRESHOLD).sum()), int((y >= THRESHOLD).sum())],
        'eligible_issues_2025_q1_2026': groups, 'threshold': THRESHOLD,
        'seconds': time.monotonic() - started,
    })
    print((RUN / 'preparation.json').read_text('utf-8'), flush=True)


def passes(candidate, baseline, sufficient_support):
    return bool(sufficient_support and candidate['coverage'] == 1
                and candidate['high_mae'] <= .97 * baseline['high_mae']
                and candidate['recall'] >= baseline['recall']
                and candidate['mae'] <= 1.05 * baseline['mae']
                and candidate['ordinary_mae'] <= 1.05 * baseline['ordinary_mae']
                and candidate['false_positive_rate'] <= baseline['false_positive_rate'] + .01)


def score_window(bounds, rounds):
    issues, x, truth, values, _ = FrozenM1().grid(bounds, [bounds])
    pins = {}
    for number in rounds:
        model_path = RUN / f'round{number}.txt'
        receipt = read(RUN / f'round{number}.json')
        pins[str(number)] = sha(model_path)
        if (receipt['model_sha256'] != pins[str(number)] or receipt['round'] != number
                or receipt['inputs_sha256'] != read(RUN / 'preparation.json')['matrix_sha256']
                or receipt['worker_sha256'] != sha(HERE / 'gpu_fit.py')
                or receipt['device'] != 'gpu' or receipt['fit_seconds'] > 600):
            raise ValueError('GPU training receipt does not match the expected experiment')
        model = lgb.Booster(model_file=str(model_path))
        if model.feature_name() != FEATURES:
            raise ValueError('Correction feature order mismatch')
        delta = model.predict(x, num_threads=2).reshape(truth.shape)
        values[f'round{number}'] = np.maximum(0, values['M1'] + np.clip(delta, -15, 15))
    high = np.isfinite(truth) & (truth >= THRESHOLD)
    target_hours = issues.repeat(24) + pd.to_timedelta(np.tile(HORIZONS - 1, len(issues)), unit='h')
    high_hours = target_hours[high.ravel()].unique()
    high_days = high_hours.normalize().nunique()
    metrics = {}
    for name, predictions in values.items():
        result = diagnostic(predictions, truth, THRESHOLD, values['persistence'][:, 0])
        result['late_mae'] = float(np.nanmean(np.nanmean(np.abs(predictions[:, 12:] - truth[:, 12:]), axis=0)))
        metrics[name] = result
    metrics_supported = len(high_hours) >= 24 and high_days >= 5 and 'ordinary_mae' in metrics['M1']
    feasible = {str(n): passes(metrics[f'round{n}'], metrics['M1'], metrics_supported) for n in rounds}
    good = [n for n in rounds if feasible[str(n)]]
    selected = min(good, key=lambda n: (metrics[f'round{n}']['high_mae'], metrics[f'round{n}']['mae'], n)) if good else None
    grid = pd.DataFrame({'issue_time': issues.repeat(24), 'horizon_hours': np.tile(HORIZONS, len(issues)),
                         'truth': truth.ravel()} | {name: v.ravel() for name, v in values.items()})
    return {
        'bounds': bounds, 'eligible_issues': len(issues), 'high_target_hours': len(high_hours),
        'high_target_days': high_days, 'metrics': metrics, 'passes': feasible,
        'selected': selected, 'model_sha256': pins,
    }, grid


def evaluate(stage):
    path = RUN / f'{stage}.json'
    if path.exists():
        raise FileExistsError(path)
    if stage == 'validation':
        bounds, rounds = VALIDATION, (1, 2)
    else:
        validation = read(RUN / 'validation.json')
        selected = validation['selected']
        if selected not in (1, 2) or sha(RUN / f'round{selected}.txt') != validation['model_sha256'][str(selected)]:
            raise ValueError('Holdout requires a frozen candidate that passed validation')
        bounds, rounds = HOLDOUT, (selected,)
        write_new(RUN / 'holdout_started.json', {'selected': selected, 'bounds': bounds,
                                                'validation_sha256': sha(RUN / 'validation.json')})
    result, grid = score_window(bounds, rounds)
    grid.to_parquet(RUN / f'{stage}_predictions.parquet', index=False)
    write_new(path, result)
    print(path.read_text('utf-8'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'validation', 'holdout'))
    stage = parser.parse_args().stage
    prepare() if stage == 'prepare' else evaluate(stage)
