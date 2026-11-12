"""Calibrate C25's newly raised alerts on frozen features."""

import sys
from pathlib import Path
from time import monotonic

HERE = Path(__file__).resolve().parent
RESIDUAL = HERE.parent / 'r140_residual'
sys.path.insert(0, str(RESIDUAL))

import train as parent
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np, pd, lgb = parent.np, parent.pd, parent.lgb
Q = parent.THRESHOLD
RUN = HERE / 'run'
FEATURES = [
    'c25', 'uplift', 'old_probability', 'local_latest', 'local_change3',
    'peer_high_fraction', 'peer_high_fraction_change3', 'horizon_hours',
]
QUANTILES = {'C25-G60': .60, 'C25-G75': .75}


def grid(frozen, round1, r140, c25, bounds):
    issues, x, truth, values = parent.baseline_grid(frozen, round1, r140, bounds)
    base = values['R1-40'].ravel()
    c25_value = parent.apply(c25, x, values['R1-40'])
    values['C25'] = c25_value
    corrected = c25_value.ravel()
    uplift = corrected - base
    inputs = np.column_stack([
        corrected, uplift, x[:, [parent.FEATURES.index(name) for name in FEATURES[2:]]],
    ])
    issue = issues.repeat(24)
    horizon = np.tile(parent.HORIZONS, len(issues))
    target_hour = issue + pd.to_timedelta(horizon - 1, unit='h')
    eligible = (base < Q) & (uplift > 0.) & (corrected >= Q - parent.CAP)
    crosses = (base < Q) & (corrected >= Q)
    return issue, horizon, target_hour, inputs, eligible, crosses, truth, values


def fit(x, y, hours):
    if not y.any() or y.all():
        raise ValueError('Both threshold classes are required')
    counts = pd.Series(hours).value_counts()
    weights = 1. / counts.reindex(hours).to_numpy()
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x = imputer.fit_transform(x)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    clf = LogisticRegression(C=.25, max_iter=300, solver='lbfgs')
    clf.fit(x, y, sample_weight=weights)
    state = dict(
        medians=imputer.statistics_.tolist(), means=scaler.mean_.tolist(),
        scales=scaler.scale_.tolist(), coefficients=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
    )
    np.testing.assert_allclose(expit(x @ state['coefficients'] + state['intercept']),
                               clf.predict_proba(x)[:, 1], rtol=0, atol=1e-12)
    return state


def probability(model, x):
    filled = np.where(np.isnan(x), model['medians'], x)
    scaled = (filled - model['means']) / model['scales']
    return expit(scaled @ model['coefficients'] + model['intercept'])


def train(data):
    _, _, hours, x, eligible, crosses, truth, _ = data
    labelled = np.isfinite(truth.ravel())
    y = truth.ravel() >= Q
    learn = eligible & labelled
    month = hours.month.to_numpy()
    oof = np.full(len(y), np.nan)
    folds = []
    for held_month in (4, 5, 6):
        fit_rows = learn & (month != held_month)
        held_rows = crosses & labelled & (month == held_month)
        if not held_rows.any():
            raise ValueError('Expected eligible alert rows in each calibration month')
        state = fit(x[fit_rows], y[fit_rows], hours[fit_rows])
        oof[held_rows] = probability(state, x[held_rows])
        folds.append({'month': held_month, 'fit_rows': int(fit_rows.sum()), 'alert_rows': int(held_rows.sum())})
    ordinary_alerts = oof[crosses & labelled & ~y]
    if not len(ordinary_alerts) or not np.isfinite(ordinary_alerts).all():
        raise ValueError('Incomplete out-of-fold ordinary alert probabilities')
    thresholds = {name: float(np.quantile(ordinary_alerts, q, method='higher')) for name, q in QUANTILES.items()}
    model = fit(x[learn], y[learn], hours[learn])
    return {
        'features': FEATURES, 'fit_bounds': parent.VALIDATION, 'model': model, 'thresholds': thresholds,
        'fit_rows': int(learn.sum()), 'high_fit_rows': int((learn & y).sum()),
        'unique_fit_target_hours': hours[learn].nunique(),
        'oof_ordinary_alerts': len(ordinary_alerts), 'folds': folds,
    }


def evaluate(data, state):
    issue, horizon, hours, x, _, crosses, truth, values = data
    base = values['R1-40'].ravel()
    c25 = values['C25'].ravel()
    alert_p = probability(state['model'], x[crosses])
    if not np.isfinite(alert_p).all():
        raise ValueError('Incomplete alert probabilities')
    for name, threshold in state['thresholds'].items():
        if not 0 < threshold < 1:
            raise ValueError('Invalid learned alert threshold')
        prediction = c25.copy()
        rejected = np.flatnonzero(crosses)[alert_p < threshold]
        retained = alert_p[alert_p < threshold] / threshold
        prediction[rejected] = np.minimum(np.nextafter(float(Q), -np.inf),
                                          base[rejected] + (Q - base[rejected]) * retained)
        values[name] = prediction.reshape(truth.shape)
    high_hours = hours[truth.ravel() >= Q].unique()
    support = len(high_hours) >= 24 and high_hours.normalize().nunique() >= 5
    current = values['persistence'][:, 0]
    metrics = {name: parent.diagnostic(pred, truth, Q, current) for name, pred in values.items()}
    old, improved = metrics['R1-40'], metrics['C25']
    recall_floor = old['recall'] + .50 * (improved['recall'] - old['recall'])
    high_error_cap = old['high_mae'] - .75 * (old['high_mae'] - improved['high_mae'])
    gates = {}
    for name in QUANTILES:
        m = metrics[name]
        gates[name] = {
            'support': support,
            'false_positive_rate': m['false_positive_rate'] <= old['false_positive_rate'] + .0025,
            'recall': m['recall'] >= recall_floor,
            'high_mae': m['high_mae'] <= high_error_cap,
            'overall_mae': m['mae'] <= 1.02 * old['mae'],
            'ordinary_mae': m['ordinary_mae'] <= 1.02 * old['ordinary_mae'],
            'original_m1_guard': parent.passes(m, metrics['M1'], support),
        }
    good = [name for name in gates if all(gates[name].values())]
    selected = min(good, key=lambda n: (-metrics[n]['recall'], metrics[n]['high_mae'],
                                        metrics[n]['false_positive_rate'])) if good else None
    rows = pd.DataFrame({'issue_time': issue, 'horizon_hours': horizon, 'truth': truth.ravel()}
                        | {name: pred.ravel() for name, pred in values.items()})
    rows.to_parquet(RUN / 'replay_predictions.parquet', index=False)
    parent.write_new(RUN / 'replay.json', {
        'status': 'reused_development', 'bounds': parent.HOLDOUT, 'selected': selected,
        'high_target_hours': len(high_hours), 'metrics': metrics, 'gates': gates,
        'recall_floor': recall_floor, 'high_error_cap': high_error_cap,
        'classifier_sha256': parent.sha(RUN / 'classifier.json'),
    })
    print('Selected', selected, flush=True)


def main():
    RUN.mkdir(exist_ok=False)
    prior = parent.read(RESIDUAL / 'run/execution.json')
    pins = parent.read(RESIDUAL / 'run/replay.json')['model_sha256']
    r140_path = HERE.parent / 'round1_refinement/run/R1-40.txt'
    c25_path = RESIDUAL / 'run/R140-C25.txt'
    round1, round1_pin = parent.parent_model()
    if (round1_pin != prior['round1_sha256'] or parent.sha(r140_path) != prior['r140_sha256']
            or parent.sha(c25_path) != pins['R140-C25']):
        raise ValueError('Frozen parent hash mismatch')
    parent.write_new(RUN / 'execution.json', {
        'protocol_sha256': parent.sha(HERE / 'PROTOCOL.md'), 'source_sha256': parent.sha(__file__),
        'parent_sha256': pins['R140-C25'], 'r140_sha256': prior['r140_sha256'],
        'round1_sha256': round1_pin, 'maximum_fits': 4,
    })
    frozen = parent.FrozenM1()
    r140 = lgb.Booster(model_file=str(r140_path))
    c25 = lgb.Booster(model_file=str(c25_path))
    data = grid(frozen, round1, r140, c25, parent.VALIDATION)
    started = monotonic()
    state = train(data)
    state['training_seconds'] = monotonic() - started
    parent.write_new(RUN / 'classifier.json', state)
    print('Trained on', state['fit_rows'], 'rows; cutoffs', state['thresholds'], flush=True)
    evaluate(grid(frozen, round1, r140, c25, parent.HOLDOUT), parent.read(RUN / 'classifier.json'))


if __name__ == '__main__':
    main()
