# Data

`observations.npz` contains the 23 released January/April conditional dates, their positions in the 86-day interval, diagonal/off-diagonal observations, and squared-error background terms for unsupported OD pairs. Those background terms retain the official dense evaluation denominator.

`calendar.npz` contains the prediction dates, node/edge IDs, the public recovery curve, and Train269 off-diagonal observations used for the frozen prior. `ReBOUND/config/protocol.json` lists the exact Train269 and Val23 dates.

`training/experts/` contains masked periodicity inputs, grouped-expert sampling views, and target-task foundation contexts. The target task uses Train269. The three additional source tasks are historical pseudo-gaps used to construct residual supervision; the individual expert workers also support these inputs.

`training/residual/` contains the prepared feature/target matrices for seven shared heads and 24 conditional ensemble fits, plus the residual matrix for fitting the low-rank bases. The ensemble matrices preserve the nested date exclusions used to construct them. They are training features and targets, not fitted trees. `training/routing.npz` contains Train-only normalization statistics.

There are no February–March target labels. The provided arrays are derived from the HuMob 2026 challenge data and let the released commands run without paths to the original workspace. Rebuilding these prepared cross-fitting matrices from the raw challenge dataset is outside the supplied training command.
