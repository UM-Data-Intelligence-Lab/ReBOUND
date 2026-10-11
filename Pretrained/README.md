# Pretrained artifacts

| Directory | Contents |
|---|---|
| `experts/` | Periodicity and foundation-model trajectories, grouped adjustment ratios, and spatial proposals |
| `routing/` | Expert choices, fitted convex weights, and calibration settings |
| `residual/` | Residual bases, shared trees, and models for 24 conditional scopes |

In `residual/`, `shared.joblib` contains seven shared gradient-boosted regressors. Each `scope_00.joblib` through `scope_23.joblib` contains a boundary regressor and two ensembles of 16 regressors. Scopes 0–22 exclude one conditional date each; scope 23 is used for deployment. `basis.npz` contains the residual bases and fixed feature statistics, and `recipe.json` defines the order and strength of the residual updates.

`train.py --reproduction true` uses these artifacts for CPU reproduction. With `--reproduction false`, it trains the periodicity/grouped networks, routers, bases, and residual trees into a separate output directory. Foundation and spatial features are retained unless regenerated Chronos outputs are supplied with `--foundation-dir`.

The spatial proposals retain the weather-conditioned branch. Its original weather-provider download URL was not recorded. Evacuation and water-recovery inputs originated from [Ishikawa disaster reports](https://www.pref.ishikawa.lg.jp/saigai/202401jishin.html); no external data is fetched during pretrained reproduction.
