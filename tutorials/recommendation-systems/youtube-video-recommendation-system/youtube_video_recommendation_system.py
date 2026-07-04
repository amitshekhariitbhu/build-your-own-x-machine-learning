import numpy as np


class YouTubeRecommender:
    """Implicit-feedback video recommender trained with Bayesian Personalized
    Ranking (BPR). Learns user/video latent factors so that watched videos are
    ranked above unwatched ones, then recommends the top-N unseen videos."""

    def __init__(self, n_factors=24, lr=0.05, reg=0.002, epochs=40, batch=512):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch = batch

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, pairs, n_users, n_items):
        # pairs: (N, 2) array of observed (user, watched_video) interactions.
        self.n_users, self.n_items = n_users, n_items
        rng = np.random.RandomState(1)

        # Small random latent factors for users (P) and videos (Q).
        self.P = 0.1 * rng.randn(n_users, self.n_factors)
        self.Q = 0.1 * rng.randn(n_items, self.n_factors)

        # Boolean watch matrix for fast "is this a true negative?" checks.
        self.seen = np.zeros((n_users, n_items), dtype=bool)
        self.seen[pairs[:, 0], pairs[:, 1]] = True

        users, pos = pairs[:, 0], pairs[:, 1]
        n = len(users)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                u, i = users[idx], pos[idx]

                # Sample negatives, resampling any that the user actually watched.
                j = rng.randint(0, n_items, size=len(idx))
                bad = self.seen[u, j]
                while bad.any():
                    j[bad] = rng.randint(0, n_items, size=int(bad.sum()))
                    bad = self.seen[u, j]

                pu, qi, qj = self.P[u], self.Q[i], self.Q[j]
                # BPR: maximize log-sigmoid of (score_pos - score_neg).
                x = np.sum(pu * (qi - qj), axis=1)
                g = self._sigmoid(-x)[:, None]  # gradient weight

                gpu = g * (qi - qj) - self.reg * pu
                gqi = g * pu - self.reg * qi
                gqj = -g * pu - self.reg * qj

                # np.add.at accumulates correctly when indices repeat in a batch.
                np.add.at(self.P, u, self.lr * gpu)
                np.add.at(self.Q, i, self.lr * gqi)
                np.add.at(self.Q, j, self.lr * gqj)
        return self

    def predict(self, user):
        # Affinity scores for every video for one user.
        return self.P[user] @ self.Q.T

    def recommend(self, user, n=10, exclude=None):
        scores = self.predict(user)
        if exclude is not None:
            scores[list(exclude)] = -np.inf  # hide already-watched videos
        return np.argsort(scores)[::-1][:n]


def make_data(n_users=400, n_items=250, n_topics=8, seed=0):
    """Synthetic YouTube watch log with planted interest structure: each user
    likes a couple of topics, each video belongs to one topic, and watches are
    sampled preferentially from a user's topics -> recoverable low-rank signal."""
    rng = np.random.RandomState(seed)

    # Each user gets 1-2 strong topic interests.
    user_int = np.zeros((n_users, n_topics))
    for u in range(n_users):
        for t in rng.choice(n_topics, size=rng.randint(1, 3), replace=False):
            user_int[u, t] = rng.uniform(0.7, 1.0)

    video_topic = rng.randint(0, n_topics, size=n_items)
    video_int = np.eye(n_topics)[video_topic]

    # Watch probability grows with user<->video topic affinity.
    affinity = user_int @ video_int.T
    prob = YouTubeRecommender._sigmoid(6.0 * (affinity - 0.55))
    watched = rng.rand(n_users, n_items) < prob

    pairs = np.argwhere(watched)
    return pairs, n_users, n_items, video_topic


if __name__ == "__main__":
    np.random.seed(0)
    pairs, n_users, n_items, video_topic = make_data()

    # Hold out one watched video per user (with >=2 watches) for evaluation.
    rng = np.random.RandomState(2)
    by_user = {}
    for u, i in pairs:
        by_user.setdefault(u, []).append(i)

    test = {}
    train_pairs = []
    for u, items in by_user.items():
        if len(items) >= 2:
            held = rng.choice(items)
            test[u] = held
            train_pairs += [(u, i) for i in items if i != held]
        else:
            train_pairs += [(u, i) for i in items]
    train_pairs = np.array(train_pairs)

    model = YouTubeRecommender().fit(train_pairs, n_users, n_items)

    # Metrics on held-out videos: AUC (rank of held-out vs unwatched) + Recall@K.
    K = 10
    aucs, hits, topic_hits = [], [], []
    for u, held in test.items():
        seen_train = set(np.where(model.seen[u])[0]) - {held}
        scores = model.predict(u)
        held_score = scores[held]
        cand = np.ones(n_items, dtype=bool)
        cand[list(seen_train)] = False
        cand[held] = False
        neg_scores = scores[cand]  # scores of unwatched, non-held videos
        aucs.append(np.mean(held_score > neg_scores))

        recs = model.recommend(u, n=K, exclude=seen_train)
        hits.append(held in recs)
        # Do top-K recommendations share the held-out video's topic?
        topic_hits.append(np.mean(video_topic[recs] == video_topic[held]))

    auc = np.mean(aucs)
    recall = np.mean(hits)
    topic_purity = np.mean(topic_hits)

    print("Users: {}  Videos: {}  Watches: {}".format(n_users, n_items, len(pairs)))
    print("Held-out test users: {}".format(len(test)))
    print("-" * 48)
    print("AUC (held-out watched vs unwatched): {:.3f}  [random = 0.500]".format(auc))
    print("Recall@{}:                           {:.3f}  [random = {:.3f}]".format(
        K, recall, K / n_items))
    print("Top-{} topic match to held-out video: {:.3f}  [random = {:.3f}]".format(
        K, topic_purity, 1.0 / 8))
    print("-" * 48)
    print("PASS" if auc > 0.8 and recall > 10 * (K / n_items) else "FAIL")
