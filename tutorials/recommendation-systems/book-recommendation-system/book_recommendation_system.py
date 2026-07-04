import numpy as np


class BookRecommender:
    """User-based collaborative filtering (kNN) with Pearson similarity."""

    def __init__(self, k=20, seed=0):
        self.k = k              # number of nearest neighbors to average over
        self.seed = seed

    def fit(self, R, mask):
        # R: user x book ratings, mask: 1 where a rating is observed.
        self.R = R
        self.mask = mask
        n_users = R.shape[0]

        # Per-user mean over observed ratings (for mean-centering).
        counts = mask.sum(axis=1)
        self.user_mean = np.where(counts > 0, (R * mask).sum(axis=1) / np.maximum(counts, 1), 0.0)

        # Mean-center observed ratings; unobserved stay 0 so they don't contribute.
        Rc = (R - self.user_mean[:, None]) * mask

        # Pearson similarity = cosine of mean-centered rating vectors.
        norms = np.sqrt((Rc ** 2).sum(axis=1))
        denom = np.outer(norms, norms)
        self.sim = np.where(denom > 0, (Rc @ Rc.T) / np.maximum(denom, 1e-8), 0.0)
        np.fill_diagonal(self.sim, 0.0)     # a user is not its own neighbor

        self.Rc = Rc
        # Per-book mean baseline (over observed ratings) for comparison / fallback.
        bcounts = mask.sum(axis=0)
        self.book_mean = np.where(bcounts > 0, (R * mask).sum(axis=0) / np.maximum(bcounts, 1),
                                  R[mask == 1].mean())
        return self

    def predict(self, u, i):
        # Neighbors who actually rated book i, ranked by similarity to u.
        rated = self.mask[:, i] == 1
        sims = self.sim[u] * rated
        nbrs = np.argsort(np.abs(sims))[::-1][:self.k]
        w = sims[nbrs]
        if np.abs(w).sum() < 1e-8:
            return self.book_mean[i]        # no useful neighbors -> book-mean fallback
        # Similarity-weighted average of neighbors' mean-centered ratings.
        pred = self.user_mean[u] + (w @ self.Rc[nbrs, i]) / np.abs(w).sum()
        return np.clip(pred, 1.0, 5.0)

    def predict_all(self):
        # Vectorized prediction for every (user, book) cell.
        S = self.sim                                   # user-user similarity
        weight = np.abs(S) @ self.mask                 # sum |sim| over raters of each book
        num = S @ self.Rc                              # weighted sum of centered ratings
        out = self.user_mean[:, None] + np.where(weight > 1e-8, num / np.maximum(weight, 1e-8), 0.0)
        out = np.where(weight > 1e-8, out, self.book_mean[None, :])
        return np.clip(out, 1.0, 5.0)

    def recommend(self, u, n=5):
        # Top-n books the user has NOT rated in the training set.
        scores = self.predict_all()[u]
        scores[self.mask[u] == 1] = -np.inf
        return np.argsort(scores)[::-1][:n]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted taste clusters: users belong to reader groups, books to genres,
    # and a user's rating is driven by how much their group likes each genre.
    n_users, n_books, n_groups, n_genres = 120, 80, 4, 4
    user_group = np.random.randint(0, n_groups, n_users)
    book_genre = np.random.randint(0, n_genres, n_books)
    group_taste = np.random.uniform(1.0, 5.0, size=(n_groups, n_genres))   # latent preference

    ground = group_taste[user_group][:, book_genre]                        # noise-free signal
    R_full = np.clip(ground + 0.4 * np.random.randn(n_users, n_books), 1.0, 5.0)

    # Hold out 20% of entries as a test split.
    test_mask = (np.random.rand(n_users, n_books) < 0.2).astype(int)
    train_mask = 1 - test_mask

    model = BookRecommender(k=20)
    model.fit(R_full * train_mask, train_mask)

    # Held-out RMSE vs. the per-book mean baseline.
    pred = model.predict_all()
    ti, tj = np.where(test_mask == 1)
    truth = R_full[ti, tj]
    cf_rmse = np.sqrt(np.mean((truth - pred[ti, tj]) ** 2))
    base_rmse = np.sqrt(np.mean((truth - model.book_mean[tj]) ** 2))

    # How often a user's top neighbor shares their planted reader group.
    top_nbr = np.argmax(model.sim, axis=1)
    nbr_acc = np.mean(user_group[top_nbr] == user_group)

    print("Users x Books:        %d x %d  (%d groups, %d genres)" % (n_users, n_books, n_groups, n_genres))
    print("Held-out RMSE (kNN):  %.4f" % cf_rmse)
    print("Baseline RMSE (book): %.4f" % base_rmse)
    print("Improvement:          %.1f%%" % (100 * (1 - cf_rmse / base_rmse)))
    print("Top-neighbor group match: %.2f (random %.2f)" % (nbr_acc, 1.0 / n_groups))
    print("Top-5 for user 0:     %s" % model.recommend(0, n=5).tolist())
    print("kNN beats baseline:   %s" % (cf_rmse < base_rmse))
