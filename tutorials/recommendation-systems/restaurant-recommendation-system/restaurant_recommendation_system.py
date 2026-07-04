import numpy as np


class RestaurantRecommender:
    """Hybrid recommender: learn each user's cuisine/price taste from their
    ratings, then score = content match + restaurant quality - price penalty."""

    def __init__(self, alpha=1.0, beta=0.5, gamma=1.0):
        self.alpha = alpha      # weight on cuisine-content match
        self.beta = beta        # weight on restaurant average rating
        self.gamma = gamma      # penalty on price mismatch

    def fit(self, R, mask, C, price, quality, loc):
        # R: user x restaurant ratings, mask: 1 where observed.
        # C: restaurant x cuisine one-hot, price/quality/loc: item features.
        self.mask, self.C = mask, C
        self.price, self.quality, self.loc = price, quality, loc

        # Center each user's ratings over their observed restaurants.
        seen = mask.sum(1, keepdims=True)
        user_mean = (R * mask).sum(1, keepdims=True) / np.maximum(seen, 1)
        Rc = (R - user_mean) * mask                     # 0 where unobserved

        # Latent cuisine taste: cuisines rated above the user's mean score high.
        self.A = Rc @ C                                 # users x cuisines

        # Latent price taste: price of restaurants the user liked (weighted mean).
        W = mask * np.clip(Rc, 0, None)                 # positive weight on liked
        denom = W.sum(1)
        self.pref_price = np.where(denom > 0, (W @ price) / np.maximum(denom, 1e-9),
                                   price.mean())
        return self

    def scores(self):
        # Full user x restaurant hybrid score matrix.
        content = self.A @ self.C.T                     # taste dotted with cuisine
        price_pen = np.abs(self.price[None, :] - self.pref_price[:, None])
        return (self.alpha * content + self.beta * self.quality[None, :]
                - self.gamma * price_pen)

    def recommend(self, u, n=5, max_dist=None):
        # Top-n restaurants the user has not visited, optionally within max_dist.
        s = self.scores()[u].copy()
        s[self.mask[u] == 1] = -np.inf                  # hide already-visited
        if max_dist is not None:                        # optional location filter
            d = np.linalg.norm(self.loc - self.loc_of_user(u), axis=1)
            s[d > max_dist] = -np.inf
        return np.argsort(s)[::-1][:n]

    def loc_of_user(self, u):
        return self._user_loc[u]


if __name__ == "__main__":
    np.random.seed(0)

    n_cuisines, n_rest, n_users = 8, 200, 150

    # Restaurant features. Average rating (quality) is planted INDEPENDENT of
    # cuisine, so a rating-only baseline carries no cuisine signal.
    cuisine_id = np.random.randint(0, n_cuisines, n_rest)
    C = np.eye(n_cuisines)[cuisine_id]                  # one-hot cuisine
    price = np.random.rand(n_rest)                      # price level in [0, 1]
    loc = np.random.rand(n_rest, 2)                     # 2-D location
    quality = np.clip(3.0 + 0.8 * np.random.randn(n_rest), 1, 5)

    # Users with a planted favorite cuisine and preferred price point.
    pref_cuisine = np.random.randint(0, n_cuisines, n_users)
    pref_price = np.random.rand(n_users)
    user_loc = np.random.rand(n_users, 2)

    # True rating = quality + big bonus for the user's cuisine - price mismatch.
    match = (cuisine_id[None, :] == pref_cuisine[:, None]).astype(float)
    price_pen = np.abs(price[None, :] - pref_price[:, None])
    R_full = np.clip(quality[None, :] + 1.8 * match - 1.2 * price_pen
                     + 0.3 * np.random.randn(n_users, n_rest), 1, 5)

    # Each user visits a random subset; the rest stay fully unseen (held out).
    observed = np.zeros((n_users, n_rest))
    for u in range(n_users):
        observed[u, np.random.choice(n_rest, 30, replace=False)] = 1.0

    model = RestaurantRecommender(alpha=1.0, beta=0.5, gamma=1.0)
    model.fit(R_full, observed, C, price, quality, loc)
    model._user_loc = user_loc

    # Evaluate on UNSEEN restaurants: does the top-K carry the right cuisine,
    # and does it beat recommending purely by average rating?
    K = 10
    hyb_cui, base_cui, hyb_rat, base_rat = 0.0, 0.0, 0.0, 0.0
    S = model.scores()
    for u in range(n_users):
        rating_only = quality.copy()                   # baseline: ignore taste
        rating_only[observed[u] == 1] = -np.inf
        s = S[u].copy()
        s[observed[u] == 1] = -np.inf

        h = np.argsort(s)[::-1][:K]                     # hybrid top-K (unseen)
        b = np.argsort(rating_only)[::-1][:K]           # baseline top-K (unseen)
        hyb_cui += np.mean(cuisine_id[h] == pref_cuisine[u])
        base_cui += np.mean(cuisine_id[b] == pref_cuisine[u])
        hyb_rat += np.mean(R_full[u, h])                # true satisfaction
        base_rat += np.mean(R_full[u, b])

    hyb_cui, base_cui = hyb_cui / n_users, base_cui / n_users
    hyb_rat, base_rat = hyb_rat / n_users, base_rat / n_users
    rand_cui = 1.0 / n_cuisines

    print("Users x Restaurants:      %d x %d  (%d cuisines)" % (n_users, n_rest, n_cuisines))
    print("Cuisine-match@%d (hybrid): %.4f" % (K, hyb_cui))
    print("Cuisine-match@%d (rating): %.4f" % (K, base_cui))
    print("Cuisine-match@%d (random): %.4f" % (K, rand_cui))
    print("Lift over random:         %.1fx" % (hyb_cui / rand_cui))
    print("Avg true rating (hybrid): %.4f" % hyb_rat)
    print("Avg true rating (rating): %.4f" % base_rat)
    print("Sample recs for user 0:   %s" % model.recommend(0, n=5).tolist())
    print("Nearby recs (dist<0.3):   %s" % model.recommend(0, n=5, max_dist=0.3).tolist())
    print("Hybrid beats baseline:    %s" % (hyb_cui > base_cui and hyb_rat > base_rat))
