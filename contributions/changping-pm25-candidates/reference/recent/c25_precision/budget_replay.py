"""Compare label-free forecast alert budgets using saved, frozen risk scores."""

from train_precision import HERE, RUN, alert_metrics, base, np, pd


def budget_alerts(frame):
    keys = frame[['issue_time', 'horizon_hours']]
    if keys.duplicated().any() or not keys.groupby('issue_time').size().eq(24).all():
        raise ValueError('Expected one complete trajectory per issue')
    if not frame.issue_time.is_monotonic_increasing:
        raise ValueError('Expected issue order for past-only budgets')
    by_issue = frame.issue_time.iloc[::24].reset_index(drop=True)
    score = frame.tree_probability.to_numpy().reshape(-1, 24)
    concentration = frame.C25_concentration.to_numpy().reshape(score.shape)
    horizon = frame.horizon_hours.to_numpy().reshape(score.shape)
    counts = frame.C25_alarm.to_numpy().reshape(score.shape).sum(axis=1)
    same_issue = np.zeros_like(score, dtype=bool)
    past_week = same_issue.copy()
    for i, issued in enumerate(by_issue):
        order = np.lexsort((horizon[i], -concentration[i], -score[i]))
        same_issue[i, order[:counts[i]]] = True
        start = by_issue.searchsorted(issued - pd.Timedelta(hours=168))
        k = counts[start:i].sum()
        if i - start < 72 or k == 0:
            past_week[i] = same_issue[i]
            continue
        previous = score[start:i].ravel()
        cutoff = np.partition(previous, len(previous) - k)[len(previous) - k]
        past_week[i] = score[i] >= cutoff
    if not np.array_equal(same_issue.sum(axis=1), counts):
        raise ValueError('Per-issue warning count drifted')
    return {'tree-same-issue': same_issue.ravel(), 'tree-past-week': past_week.ravel()}


def main():
    frame = pd.read_parquet(RUN / 'replay_predictions.parquet')
    frozen = base.read(RUN / 'replay.json')
    if frozen['fits_sha256'] != base.sha(RUN / 'fits.json'):
        raise ValueError('Probability source receipt changed')
    decision = budget_alerts(frame)
    prefix = (len(frame) // 48) * 24
    changed = frame.copy()
    changed.loc[prefix:, 'tree_probability'] = 0.123
    changed.loc[prefix:, 'C25_alarm'] = False
    for name, predicted in budget_alerts(changed).items():
        np.testing.assert_array_equal(predicted[:prefix], decision[name][:prefix])
    scored = np.isfinite(frame.truth.to_numpy())
    high = frame.truth.to_numpy()[scored] >= 73
    metrics = {name: alert_metrics(value[scored], high)
               for name, value in ({'C25': frame.C25_alarm.to_numpy()} | decision).items()}
    baseline = metrics['C25']
    passed = {
        name: m['precision'] > baseline['precision'] and m['recall'] >= .70
        and m['false_positive_rate'] < baseline['false_positive_rate']
        for name, m in metrics.items() if name != 'C25'
    }
    eligible = [name for name, ok in passed.items() if ok]
    chosen = max(eligible, key=lambda n: (metrics[n]['precision'], metrics[n]['recall'])) if eligible else None
    frame[['issue_time', 'horizon_hours', 'truth', 'C25_alarm']].assign(**decision).to_parquet(
        RUN / 'budget_predictions.parquet', index=False)
    base.write_new(RUN / 'budget.json', {
        'bounds': frozen['bounds'], 'status': 'reused_development', 'selected': chosen,
        'metrics': metrics, 'passes': passed, 'protocol_sha256': base.sha(HERE / 'BUDGET_PROTOCOL.md'),
        'future_perturbation_prefix_rows': prefix,
    })
    print(metrics, passed, 'selected', chosen)


if __name__ == '__main__':
    main()
