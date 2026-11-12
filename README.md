# bj-air-quality-forecast

Short-term PM2.5 forecasting for **35 monitoring stations in Beijing** at lead times of
**h = 1 / 3 / 6 hours**, with a frozen evaluation protocol, two persistence baselines, and an
interactive demo dashboard.

The core claim is deliberately conservative: every number below is measured against a
**persistence baseline on an identical, pre-frozen sample set**, with daily-block bootstrap
confidence intervals — not against a hand-tuned straw man.

| lead time | model MAE (µg/m³) | persistence B1 MAE | improvement | 95% CI | significant |
|---|---|---|---|---|---|
| h = 1 | 2.802 | 3.228 | **+13.21%** | +11.11% – +15.00% | ✅ |
| h = 3 | **5.965** | 7.425 | **+19.66%** | +18.27% – +21.00% | ✅ |
| h = 6 | **8.588** | 11.592 | **+25.91%** | +24.04% – +27.84% | ✅ |

*Test split ≥ 2025-07-01, LightGBM v4 (52 features, ERA5 meteorology + Jing-Jin-Ji neighbourhood
signals), 2000 bootstrap resamples over **Beijing calendar days** rather than rows — hourly PM2.5
has 1-lag autocorrelation 0.974, so row-level resampling would narrow the CI by an order of
magnitude and manufacture false precision.*

---

## What makes this not a toy

**1. A frozen protocol, not a sliding evaluation.**
`PROTOCOL.md` fixes the issue time `T` (top of hour), the information cutoff (`T-1`, i.e.
`[T-1, T)`), and the target (`[T+h-1, T+h)` — a *single* hour, not a rolling curve). The sample
set is materialised once into `data/index/protocol=p1/` with a SHA-256 fingerprint; every model
reuses the exact same `(station, issue_time, h)` rows. Nobody gets to re-filter the test set.

**2. Leakage was hunted, not assumed away.**
Section 11.1 of `RESULTS.md` documents a data-poisoning trap that had to be removed first;
section 11.3 shows that most of meteorology's benefit comes from *future-hour* weather inputs,
which is a legitimate forecast-only signal and is labelled as such.

**3. Failures are reported as loudly as gains.**
At h = 6 the model trades recall for precision (precision 0.578 → **0.724**, recall 0.588 → 0.552):
it is conservative about calling peaks, so it misses some. Heavy-pollution episodes are
systematically **underestimated**, and that is stated rather than smoothed over.

**4. A deep-learning route that was evaluated and rejected on the evidence.**
LSTM v2 (14 channels, SEQ = 24) reaches 2.846 / 6.265 / 9.425 — statistically tied with LightGBM at
h = 1 (+0.007 MAE) but behind at h = 3 / h = 6. TCN underperforms both. The single-model + threshold
route won; details in `M5_深度学习结果.md`.

---

## Architecture

```
raw daily files (quotsoft.net, official annual bundles)
        │  scripts/fetch_quotsoft.py, fetch_meteo.py, extract_zips.py
        ▼
Parquet warehouse  data/warehouse/  (20.4M rows, 189 MB compressed; raw 9.9 GB)
        │  scripts/build_long.py, build_stations.py, build_meteo.py
        ▼
frozen protocol index  data/index/protocol=p1/   ← SHA-256 fingerprint, never rebuilt
        │  scripts/build_index.py (defines B1 = y(T-1), B2 = y(target - 24h))
        ▼
feature layer  data/features/protocol=p1/h={1,3,6}/
        │  scripts/build_features.py, build_meteo_features.py, build_nb_features.py
        ▼
LightGBM  scripts/train_lgbm.py  →  data/runs/{run_id}/  (metrics, station MAE, importance, run.json)
        │  scripts/train_dl.py, train_dl2.py (LSTM / TCN, GPU)
        ▼
evaluation  scripts/evaluate.py  (daily-block bootstrap, peak detection, concentration bins)
        │  scripts/experiment_ablation.py, experiment_weight.py, experiment_heavy.py
        ▼
serving  scripts/predict_live.py (leak-free live path)  →  app/ (FastAPI + single-page Vue 3)
```

### Data scope

| dataset | coverage | volume |
|---|---|---|
| Beijing stations (`beijing_all`) | 2013-12-06 → 2026-10-29, 4711 days | 35 stations, 3,892,140 station-hours (**98.35%** availability) |
| Jing-Jin-Ji sites (`china_sites`) | 2014-05-13 → 2026-10-24 | 130 stations, 11,355,036 rows |
| Prefecture city means (`china_cities`) | 2014-05-13 → 2026-10-29 | 13 cities, hourly |
| ERA5 via Open-Meteo (`meteo`) | 2013-12 → 2026-10, incremental | 22 grid points × 11 variables |

Full field definitions, quality metrics and known defects: `DATA.md`.

---

## Repository layout

```
src/aircast/          protocol definitions, metrics (importable package)
scripts/              data build → features → training → evaluation → live inference
app/                  FastAPI service + static dashboard (overview / trend / replay / compare / live)
data/experiments/     per-experiment metric CSVs (ablation, seeds, meteorology, heavy-episode)
data/reports/         QC summary and detected data gaps
核验依据/              verification records for the data samples
PROTOCOL.md           time contract, baselines, split policy, fingerprint
RESULTS.md            all experiments, including negative results and open questions
DATA.md               data dictionary and quality report
M5_深度学习结果.md      LSTM / TCN results and the Jing-Jin-Ji ablation
北京空气质量短期预测项目指导书.md   full project specification
```

**Not in this repository** (by design, see `.gitignore`): raw and intermediate data (`data/raw`,
`data/warehouse`, `data/features`, ~14 GB), training logs, and the cluster-access helper script.
Clone and run the fetch scripts to rebuild the data layer.

---

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt
export PYTHONPATH=src
```

Rebuild from raw files (requires the raw daily downloads in `data/raw/`):

```bash
python scripts/build_long.py          # raw → long-format warehouse
python scripts/build_stations.py      # station metadata + coordinates
python scripts/build_index.py         # freeze protocol p1 index and both baselines
python scripts/build_features.py      # feature tables per lead time
python scripts/train_lgbm.py --seed 42 --tag v4
python scripts/evaluate.py --run data/runs/<run_id>
```

Live inference (independent of the frozen index, self-checks consistency with it):

```bash
python scripts/predict_live.py --time "2026-10-31 23:00"
```

Dashboard:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints: `/api/stations`, `/api/overview`, `/api/trend`, `/api/replay`, `/api/comparison`.

---

## Ablations worth knowing

| what was tested | h = 1 | h = 3 | h = 6 |
|---|---|---|---|
| Jing-Jin-Ji neighbourhood features (MAE reduction) | 2.37% | 5.86% | 7.08% |
| ERA5 meteorology (percentage points gained vs. no-meteo v3) | −0.58 | **+4.5** | **+9.3** |
| class weighting 3× mid / 10× heavy (chosen trade-off) | keep 2×5× | use 3×10× | use 3×10× |

The longer the lead time, the more the spatial and meteorological context is worth — which is the
intuitive result, and here it is also the measured one.

---

## Honest limitations

- The historical evaluation uses **reanalysis** meteorology, not the forecast that was actually
  available at issue time. Replacing it with `historical-forecast-api` is the next step and will
  lower the headline numbers; the current figures are an upper bound.
- Coordinates for some stations come from OpenStreetMap geocoding (`coord_source=nominatim_osm`)
  and have not yet been reconciled against the official station table.
- Hadoop/Spark migration is designed but **not yet executed**; the current pipeline is single-node
  pandas + LightGBM. The distributed version is tracked as future work rather than claimed.
- The prototype runs on A-share rules of the real world: one city, one pollutant family. Do not
  extrapolate to other regions without re-validating the neighbourhood features.

---

## Documentation

Chinese-language documentation is the primary reference and is kept in sync:

- `PROTOCOL.md` — time contract, baselines, split policy
- `RESULTS.md` — every experiment, including the ones that failed
- `DATA.md` — data dictionary, coverage, quality defects
- `北京空气质量短期预测项目指导书.md` — full specification (PDF / Word versions alongside)
