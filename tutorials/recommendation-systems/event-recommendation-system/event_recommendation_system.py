import numpy as np


class EventRecommender:
    """Hybrid content-based event recommender (cold-start friendly).

    Each event is a content feature vector (category + location one-hots).
    A per-user preference vector is learned from past ratings via ridge
    regression, then upcoming (never-seen) events are scored by
    preference . features + a popularity term. Because scoring depends only
    on content features, brand-new events with zero interactions can still
    be ranked -- the cold-start case that pure collaborative filtering fails.
    """

    def __init__(self, reg=1.0, pop_weight=0.3, seed=0):
        self.reg = reg                # L2 strength for the per-user solve
        self.pop_weight = pop_weight  # weight of the popularity signal
        self.seed = seed

    def fit(self, feats, ratings, mask):
        # feats: (n_events, n_feat) past-event content vectors.
        # ratings: (n_users, n_events), mask: 1 where a rating is observed.
        n_users, n_feat = ratings.shape[0], feats.shape[1]
        I = self.reg * np.eye(n_feat)
        self.W = np.zeros((n_users, n_feat))  # learned preference vectors

        for u in range(n_users):
            obs = mask[u] == 1
            F = feats[obs]                    # features of rated events
            r = ratings[u, obs]               # this user's ratings
            # Ridge regression: w = (F^T F + reg I)^-1 F^T r  (closed form).
            self.W[u] = np.linalg.solve(F.T @ F + I, F.T @ r)
        return self

    def score(self, feats, popularity):
        # Hybrid score: content preference + normalized popularity boost.
        pop = (popularity - popularity.mean()) / (popularity.std() + 1e-9)
        return self.W @ feats.T + self.pop_weight * pop[None, :]

    def recommend(self, u, feats, popularity, n=5):
        # Top-n upcoming events for user u.
        scores = self.score(feats, popularity)[u]
        return np.argsort(scores)[::-1][:n]


if __name__ == "__main__":
    np.random.seed(0)

    n_users, n_cat, n_loc = 200, 6, 4
    n_past, n_up = 120, 80          # past (train) vs upcoming (held-out) events
    n_feat = n_cat + n_loc
    K = 5

    def make_events(m):
        cat = np.random.randint(0, n_cat, m)
        loc = np.random.randint(0, n_loc, m)
        F = np.zeros((m, n_feat))
        F[np.arange(m), cat] = 1.0                  # category one-hot
        F[np.arange(m), n_cat + loc] = 1.0          # location one-hot
        pop = np.random.rand(m)                     # popularity, cat-independent
        return F, cat, loc, pop

    past_F, past_cat, past_loc, past_pop = make_events(n_past)
    up_F, up_cat, up_loc, up_pop = make_events(n_up)

    # Planted user tastes: each user strongly prefers one category, mildly a loc.
    pref_cat = np.random.randint(0, n_cat, n_users)
    pref_loc = np.random.randint(0, n_loc, n_users)

    def utility(cat, loc, pop):
        # True liking = big category match + small location match + tiny pop.
        return (3.0 * (cat[None, :] == pref_cat[:, None])
                + 1.0 * (loc[None, :] == pref_loc[:, None])
                + 0.3 * pop[None, :])

    past_util = utility(past_cat, past_loc, past_pop)
    ratings = past_util + 0.3 * np.random.randn(n_users, n_past)   # noisy ratings

    # Each user has only rated a random handful of past events (sparse history).
    mask = (np.random.rand(n_users, n_past) < 0.35).astype(int)

    model = EventRecommender(reg=1.0, pop_weight=0.3).fit(past_F, ratings, mask)

    # Relevance on HELD-OUT upcoming events = user's planted preferred category.
    relevant = (up_cat[None, :] == pref_cat[:, None])   # (n_users, n_up)

    # Model recommendations.
    scores = model.score(up_F, up_pop)
    top = np.argsort(-scores, axis=1)[:, :K]
    hit_model = relevant[np.arange(n_users)[:, None], top]
    prec_model = hit_model.mean()
    catmatch_model = (up_cat[top] == pref_cat[:, None]).mean()

    # Popularity-only baseline: same top-K popular events for everyone.
    pop_top = np.argsort(-up_pop)[:K]
    prec_pop = relevant[:, pop_top].mean()
    catmatch_pop = (up_cat[pop_top][None, :] == pref_cat[:, None]).mean()

    rand_level = 1.0 / n_cat   # chance a random upcoming event fits the taste

    print("Users x categories:        %d x %d" % (n_users, n_cat))
    print("Past / upcoming events:    %d / %d  (upcoming are unseen)" % (n_past, n_up))
    print("Precision@%d  (model):      %.3f" % (K, prec_model))
    print("Precision@%d  (popularity): %.3f" % (K, prec_pop))
    print("Precision@%d  (random):     %.3f" % (K, rand_level))
    print("Category-match (model):    %.3f" % catmatch_model)
    print("Category-match (popular):  %.3f" % catmatch_pop)
    print("Top-%d upcoming for user 0: %s" % (K, top[0].tolist()))
    print("Model beats popularity:    %s" % (prec_model > prec_pop))
    print("Model beats random:        %s" % (prec_model > rand_level))
