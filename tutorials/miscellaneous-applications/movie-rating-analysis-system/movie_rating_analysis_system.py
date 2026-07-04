import numpy as np


class MovieRatingAnalyzer:
    """From-scratch matrix-factorization recommender for movie ratings.

    Learns latent taste vectors for users and latent feature vectors for
    movies so that an observed rating is reconstructed as
        r_hat(u, i) = mu + b_u + b_i + p_u . q_i
    where mu is the global mean, b_u / b_i are user / movie bias terms and
    p_u, q_i are K-dimensional latent factors. Trained by stochastic gradient
    descent on the L2-regularized squared error over the *observed* ratings
    only -- every gradient is written by hand. Once fit, predict() fills in
    unseen (user, movie) cells and recommend() ranks a user's unseen movies.
    """

    def __init__(self, n_factors=8, lr=0.01, reg=0.05, epochs=40, seed=0):
        self.K = n_factors
        self.lr = lr          # SGD step size
        self.reg = reg        # L2 penalty on factors and biases
        self.epochs = epochs
        self.seed = seed

    def fit(self, ratings, n_users, n_movies):
        """ratings: array of (user, movie, rating) rows (the train split)."""
        rng = np.random.RandomState(self.seed)
        u = ratings[:, 0].astype(int)
        i = ratings[:, 1].astype(int)
        r = ratings[:, 2].astype(float)

        self.mu = r.mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_movies)
        self.P = rng.normal(0, 0.1, (n_users, self.K))
        self.Q = rng.normal(0, 0.1, (n_movies, self.K))

        idx = np.arange(len(r))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for k in idx:                       # per-rating SGD update
                uk, ik = u[k], i[k]
                pred = self.mu + self.bu[uk] + self.bi[ik] + self.P[uk] @ self.Q[ik]
                e = r[k] - pred
                self.bu[uk] += self.lr * (e - self.reg * self.bu[uk])
                self.bi[ik] += self.lr * (e - self.reg * self.bi[ik])
                pu = self.P[uk].copy()
                self.P[uk] += self.lr * (e * self.Q[ik] - self.reg * self.P[uk])
                self.Q[ik] += self.lr * (e * pu - self.reg * self.Q[ik])
        return self

    def predict(self, users, movies):
        users = np.asarray(users, dtype=int)
        movies = np.asarray(movies, dtype=int)
        dot = np.sum(self.P[users] * self.Q[movies], axis=1)
        pred = self.mu + self.bu[users] + self.bi[movies] + dot
        return np.clip(pred, 1.0, 5.0)

    def recommend(self, user, seen, n_movies, top=5):
        """Rank the movies this user has NOT rated by predicted score."""
        candidates = np.array([m for m in range(n_movies) if m not in seen])
        scores = self.predict(np.full(len(candidates), user), candidates)
        order = np.argsort(-scores)
        return candidates[order[:top]]


def make_ratings(n_users=200, n_movies=120, k_true=3, density=0.28, seed=0):
    """Synthetic user-movie ratings with planted low-rank taste structure.

    Each user gets a hidden taste vector and each movie a hidden genre
    vector (both K-dim). The true rating is a global mean + user/movie bias +
    the taste-genre affinity + small noise, clipped to the 1..5 star scale.
    Only a random `density` fraction of the full grid is observed, exactly
    like a real (sparse) ratings table. Returns observed rows plus the true
    latent means so the recovery can be scored.
    """
    rng = np.random.RandomState(seed)
    mu = 3.4
    user_bias = rng.normal(0, 0.5, n_users)      # generous vs harsh raters
    movie_bias = rng.normal(0, 0.6, n_movies)    # blockbuster vs flop
    U = rng.normal(0, 1.0, (n_users, k_true))    # user tastes
    V = rng.normal(0, 1.0, (n_movies, k_true))   # movie genre loadings

    true = mu + user_bias[:, None] + movie_bias[None, :] + (U @ V.T) * 0.6
    true = np.clip(true, 1.0, 5.0)

    mask = rng.rand(n_users, n_movies) < density
    us, ms = np.where(mask)
    obs = true[us, ms] + rng.normal(0, 0.35, len(us))     # observation noise
    obs = np.clip(np.round(obs * 2) / 2, 1.0, 5.0)        # half-star ratings
    ratings = np.column_stack([us, ms, obs])
    return ratings, true, n_users, n_movies


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


if __name__ == "__main__":
    np.random.seed(0)

    ratings, true, n_users, n_movies = make_ratings(seed=0)
    rng = np.random.RandomState(1)
    perm = rng.permutation(len(ratings))
    n_test = int(0.2 * len(ratings))
    test, train = ratings[perm[:n_test]], ratings[perm[n_test:]]

    model = MovieRatingAnalyzer(n_factors=8, lr=0.01, reg=0.05, epochs=40).fit(
        train, n_users, n_movies)

    tu, tm, tr = test[:, 0].astype(int), test[:, 1].astype(int), test[:, 2]
    pred = model.predict(tu, tm)

    # Baseline 1: predict every rating with the global training mean.
    gmean = train[:, 2].mean()
    base_pred = np.full(len(tr), gmean)
    # Baseline 2: predict each movie's own training-mean rating (fallback mu).
    movie_mean = np.full(n_movies, gmean)
    for m in range(n_movies):
        rows = train[train[:, 1] == m, 2]
        if len(rows):
            movie_mean[m] = rows.mean()
    mm_pred = movie_mean[tm]

    mf_rmse, base_rmse, mm_rmse = rmse(pred, tr), rmse(base_pred, tr), rmse(mm_pred, tr)

    print("=== Movie Rating Analysis System (from scratch) ===")
    print(f"users {n_users}  movies {n_movies}  ratings {len(ratings)} "
          f"(train {len(train)} / test {len(test)})")
    print(f"global-mean baseline RMSE : {base_rmse:.3f}   MAE {mae(base_pred, tr):.3f}")
    print(f"movie-mean baseline RMSE  : {mm_rmse:.3f}   MAE {mae(mm_pred, tr):.3f}")
    print(f"matrix-factorization RMSE : {mf_rmse:.3f}   MAE {mae(pred, tr):.3f}")
    print(f"improvement over global   : {(1 - mf_rmse / base_rmse) * 100:4.1f}% lower RMSE")

    # Ranking check: for a sample user, do recommended movies really rate high?
    hit = tot = 0
    for user in range(0, n_users, 5):
        seen = set(train[train[:, 0] == user, 1].astype(int))
        recs = model.recommend(user, seen, n_movies, top=5)
        liked = true[user] >= 4.0                    # ground-truth "would like"
        if liked.sum():
            hit += int(liked[recs].sum())
            tot += len(recs)
    rank_prec = hit / tot if tot else 0.0
    base_rank = float((true >= 4.0).mean())          # random-pick hit rate
    print(f"top-5 recommend precision : {rank_prec:.3f}  (random {base_rank:.3f})")

    assert mf_rmse < base_rmse - 0.05, "MF must clearly beat the global-mean baseline"
    assert mf_rmse < mm_rmse, "MF must beat the per-movie-mean baseline"
    assert rank_prec > base_rank + 0.10, "recommendations must beat random ranking"
    print("PASS: latent-factor model recovers ratings and ranks movies above baseline")
