"""Extend the existing C25 tree-sorted forecast with causal feedback."""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRECISION = HERE.parent
sys.path.insert(0, str(PRECISION))

import train_precision as previous
from budget_replay import budget_alerts

base = previous.base
np, pd, lgb, Q = previous.np, previous.pd, previous.lgb, previous.Q
RUN = HERE / 'run'
PARAMS = previous.PARAMS.copy()
FEATURES = previous.FEATURES + [
    'parent_probability', 'parent_alarm', 'previous_probability_mean',
    'previous_alarm_fraction', 'previous_available_count', 'recent_c25_bias',
]


def history_features(frame):
    issue = pd.DatetimeIndex(frame.issue_time)
    horizon = frame.horizon_hours.to_numpy()
    target = issue + pd.to_timedelta(horizon - 1, unit='h')
    keys = pd.MultiIndex.from_arrays([issue, target])
    past = frame.set_index(keys)[['tree_probability', 'parent_alarm']]
    probability = []
    alarm = []
    for lag in (1, 3, 6):
        earlier = pd.MultiIndex.from_arrays([issue - pd.Timedelta(hours=lag), target])
        found = past.reindex(earlier)
        probability.append(found.tree_probability.to_numpy())
        alarm.append(found.parent_alarm.to_numpy(dtype='float64'))
    count = np.isfinite(probability).sum(axis=0)
    mean = np.divide(np.nansum(probability, axis=0), count,
                     out=np.full(len(frame), np.nan), where=count > 0)
    agreement = np.divide(np.nansum(alarm, axis=0), count,
                          out=np.full(len(frame), np.nan), where=count > 0)
    group = (horizon - 1) // 6
    residual = frame.C25_concentration.to_numpy() - frame.truth.to_numpy()
    hourly = pd.DataFrame({'target': target, 'group': group, 'residual': residual})
    per_hour = hourly.groupby(['target', 'group']).residual.mean().unstack()
    axis = pd.date_range(target.min() - pd.Timedelta(hours=12), target.max(), freq='h')
    recent = per_hour.reindex(axis).rolling(12, min_periods=4).mean()
    bias = recent.reindex(issue - pd.Timedelta(hours=2)).to_numpy()[np.arange(len(frame)), group]
    return np.column_stack([mean, agreement, count, bias])


def make_data(frozen, round1, r140, c25, tree, bounds):
    x, truth, target, issue, horizon, c25_value = previous.matrix(frozen, round1, r140, c25, bounds)
    frame = pd.DataFrame({
        'issue_time': issue, 'horizon_hours': horizon, 'truth': truth,
        'C25_concentration': c25_value, 'C25_alarm': c25_value >= Q,
        'tree_probability': previous.predict('tree', tree, x),
    })
    frame['parent_alarm'] = budget_alerts(frame)['tree-same-issue']
    context = history_features(frame)
    x = np.column_stack([x, frame.tree_probability, frame.parent_alarm, context]).astype('float32')
    return frame, x, target


def fit(data):
    frames, arrays, targets = zip(*data)
    truth = np.concatenate([frame.truth.to_numpy() for frame in frames])
    known = np.isfinite(truth)
    x = np.concatenate(arrays)[known]
    y = truth[known] >= Q
    target = targets[0].append(targets[1])[known]
    weights = 1. / pd.Series(target).value_counts().reindex(target).to_numpy()
    tick = time.monotonic()
    model = lgb.train(PARAMS, lgb.Dataset(x, label=y, weight=weights, feature_name=FEATURES),
                      num_boost_round=100)
    path = RUN / 'feedback.txt'
    model.save_model(str(path))
    base.write_new(RUN / 'fit.json', {
        'rows': len(y), 'high_rows': int(y.sum()), 'seconds': time.monotonic() - tick,
        'weight_sha256': base.sha(path),
    })
    return lgb.Booster(model_file=str(path))


def decisions(frame, score, bias):
    concentrations = frame.C25_concentration.to_numpy()
    correction = np.clip(np.nan_to_num(bias, nan=0.) * .5, -8., 8.)
    parent_counts = frame.parent_alarm.to_numpy().reshape(-1, 24).sum(axis=1)
    adjusted_counts = ((concentrations - correction) >= Q).reshape(-1, 24).sum(axis=1)
    probabilities = score.reshape(-1, 24)
    old_probabilities = frame.tree_probability.to_numpy().reshape(-1, 24)
    horizons = frame.horizon_hours.to_numpy().reshape(-1, 24)
    order = np.lexsort((horizons, -old_probabilities, -probabilities), axis=1)
    ranks = np.argsort(order, axis=1)
    return {
        'feedback-same-count': (ranks < parent_counts[:, None]).ravel(),
        'feedback-adaptive': (ranks < adjusted_counts[:, None]).ravel(),
    }


def evaluate(data, model):
    frame, x, _ = data
    saved = pd.read_parquet(PRECISION / 'run/budget_predictions.parquet')
    keys = ['issue_time', 'horizon_hours', 'truth']
    pd.testing.assert_frame_equal(frame[keys], saved[keys], check_exact=True)
    np.testing.assert_array_equal(frame.parent_alarm, saved['tree-same-issue'])
    score = model.predict(x, num_threads=2)
    predictions = decisions(frame, score, x[:, -1])
    prefix = (len(frame) // 48) * 24
    altered = frame.copy()
    altered.loc[prefix:, ['truth', 'C25_concentration', 'tree_probability', 'parent_alarm']] = [0., 999., .5, False]
    new_context = history_features(altered)
    np.testing.assert_allclose(new_context[:prefix], history_features(frame)[:prefix], equal_nan=True)
    for name, value in decisions(altered, score, new_context[:, -1]).items():
        np.testing.assert_array_equal(value[:prefix], predictions[name][:prefix])
    known = frame.truth.notna().to_numpy()
    high = frame.truth.to_numpy()[known] >= Q
    metrics = {name: previous.alert_metrics(pred[known], high) for name, pred in
               ({'parent': frame.parent_alarm.to_numpy()} | predictions).items()}
    parent = metrics['parent']
    gates = {
        name: m['precision'] >= parent['precision'] + .01 and m['recall'] >= .70
        and m['false_positive_rate'] < parent['false_positive_rate']
        for name, m in metrics.items() if name != 'parent'
    }
    eligible = [name for name in gates if gates[name]]
    selected = max(eligible, key=lambda n: (metrics[n]['precision'], metrics[n]['recall'])) if eligible else None
    frame.assign(feedback_probability=score, recent_c25_bias=x[:, -1], **predictions).to_parquet(
        RUN / 'replay_predictions.parquet', index=False)
    base.write_new(RUN / 'replay.json', {
        'bounds': base.HOLDOUT, 'status': 'reused_development', 'metrics': metrics,
        'passes': gates, 'selected': selected, 'parent_reproduced': True,
        'future_perturbation_prefix_rows': prefix,
    })
    print(metrics, gates, selected, flush=True)


def main():
    RUN.mkdir(exist_ok=False)
    residual = previous.alert.RESIDUAL / 'run'
    receipt = base.read(residual / 'execution.json')
    c25_pin = base.read(residual / 'replay.json')['model_sha256']['R140-C25']
    r140_path = PRECISION.parent / 'round1_refinement/run/R1-40.txt'
    c25_path = residual / 'R140-C25.txt'
    tree_info = base.read(PRECISION / 'run/fits.json')['tree']
    tree_path = PRECISION / 'run' / tree_info['weights']
    round1, round1_pin = base.parent_model()
    if (round1_pin != receipt['round1_sha256'] or base.sha(r140_path) != receipt['r140_sha256']
            or base.sha(c25_path) != c25_pin or base.sha(tree_path) != tree_info['sha256']):
        raise ValueError('Frozen parent hash mismatch')
    base.write_new(RUN / 'execution.json', {
        'protocol_sha256': base.sha(HERE / 'PROTOCOL.md'), 'source_sha256': base.sha(__file__),
        'tree_sha256': tree_info['sha256'], 'c25_sha256': c25_pin, 'features': FEATURES,
    })
    frozen = base.FrozenM1()
    r140, c25, tree = (lgb.Booster(model_file=str(path)) for path in (r140_path, c25_path, tree_path))
    model = fit([make_data(frozen, round1, r140, c25, tree, period)
                 for period in (previous.ADDED, base.VALIDATION)])
    evaluate(make_data(frozen, round1, r140, c25, tree, base.HOLDOUT), model)


if __name__ == '__main__':
    main()
