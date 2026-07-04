import numpy as np

# Cricket Score Prediction Model (from scratch)
# ---------------------------------------------
# A T20 innings is a short time series: runs scored in each of 20 overs. Scoring
# ACCELERATES through the innings (steady middle overs, big hitting at the death),
# so the per-over run rate is NOT constant. We forecast the FINAL total from a
# partially played innings by decomposing each innings into two planted factors:
#   1) a shared over PROFILE  profile[o]  = typical runs in over o (the accel shape)
#   2) a per-innings STRENGTH multiplier  (how good this batting side is today)
# From the overs seen so far we estimate the team's strength, then project the
# remaining overs along the learned profile:  final = seen + strength * (rest).
# A naive "current run rate x 20" projection assumes the death overs score at the
# slow early rate, so it under-forecasts; beating it on held-out innings proves
# the model recovered the acceleration profile.


class CricketScorePredictor:
    """Over-profile x team-strength decomposition, fit from scratch."""

    def __init__(self, n_overs=20):
        self.n_overs = n_overs

    def fit(self, runs_per_over):
        R = np.asarray(runs_per_over, float)          # (n_innings, n_overs)
        self.profile = R.mean(axis=0)                 # avg runs per over = accel shape
        self.cum = np.cumsum(self.profile)            # expected cumulative by end of over o
        self.total_expected = self.cum[-1]            # expected full-innings total
        return self

    def predict(self, partial_runs, overs_seen):
        # Project the final total from the first `overs_seen` overs of each innings.
        P = np.asarray(partial_runs, float)
        seen = P[:, :overs_seen].sum(axis=1)          # runs on the board so far
        expected_seen = self.cum[overs_seen - 1]      # profile's expectation for those overs
        strength = seen / expected_seen               # this side's multiplier (>1 = strong)
        remaining = self.total_expected - expected_seen
        return seen + strength * remaining            # current runs + projected rest


def make_innings(n_innings=400, n_overs=20, seed=0):
    # Synthetic T20 scorecards: per-over runs rise through the innings (accel),
    # scaled by a per-innings batting strength, sampled as integer Poisson runs.
    np.random.seed(seed)
    over = np.arange(n_overs)
    base = 5.0 + 0.45 * over                          # run rate climbs 5 -> ~13.5
    strength = np.random.uniform(0.7, 1.3, size=n_innings)   # batting quality of the day
    lam = strength[:, None] * base[None, :]           # expected runs per (innings, over)
    runs = np.random.poisson(lam).astype(float)       # integer runs, over by over
    return runs


if __name__ == "__main__":
    np.random.seed(0)
    n_overs = 20
    runs = make_innings(n_innings=400, n_overs=n_overs)

    split = 320                                        # held-out innings for honest eval
    train, test = runs[:split], runs[split:]
    model = CricketScorePredictor(n_overs=n_overs).fit(train)

    true_final = test.sum(axis=1)                      # actual final scores (test innings)
    train_mean = train.sum(axis=1).mean()              # mean-total baseline (ignores play)

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))

    print("Cricket score prediction (over-profile x team-strength decomposition)")
    print(f"  innings={len(runs)}  train={split}  test={len(test)}  overs/innings={n_overs}")
    print(f"  learned run-rate profile (runs/over): "
          f"start={model.profile[0]:.1f}  mid={model.profile[10]:.1f}  death={model.profile[-1]:.1f}")
    print(f"  baseline train-mean total RMSE: {rmse(np.full_like(true_final, train_mean), true_final):8.3f}"
          f"   (predicts {train_mean:.1f} every time)")
    print("  forecasting the final total from a partly played innings:")
    print("    overs_seen | model RMSE | naive-rate RMSE | model MAE | naive MAE | win-rate")

    all_beat = True
    for seen in (5, 8, 10, 12, 15):
        pred = model.predict(test, seen)
        board = test[:, :seen].sum(axis=1)
        naive = board / seen * n_overs                 # baseline: current run rate x overs
        m_rmse, n_rmse = rmse(pred, true_final), rmse(naive, true_final)
        m_mae, n_mae = mae(pred, true_final), mae(naive, true_final)
        win = float(np.mean(np.abs(pred - true_final) < np.abs(naive - true_final)))
        beat = m_rmse < n_rmse and m_rmse < rmse(np.full_like(true_final, train_mean), true_final)
        all_beat = all_beat and beat
        print(f"       {seen:2d}      |  {m_rmse:8.3f}  |    {n_rmse:8.3f}     | {m_mae:7.3f}  |"
              f" {n_mae:7.3f}  |  {win:5.3f}")

    print(f"  result: {'model BEATS naive-rate + mean baselines at every checkpoint' if all_beat else 'baseline wins'}")
    print("  sample forecasts after 10 overs (held-out innings):")
    pred10 = model.predict(test, 10)
    naive10 = test[:, :10].sum(axis=1) / 10 * n_overs
    for j in range(0, len(test), max(1, len(test) // 6)):
        print(f"    at 10 ov: {test[j, :10].sum():5.0f}/board   pred={pred10[j]:6.1f}"
              f"   naive={naive10[j]:6.1f}   true={true_final[j]:6.0f}")
