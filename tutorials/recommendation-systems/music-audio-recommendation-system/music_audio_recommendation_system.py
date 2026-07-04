import numpy as np


class MusicRecommender:
    """Content-based music recommender with a small collaborative blend.
    Each track is an audio-feature vector (tempo, energy, danceability + a
    learned genre embedding). A user's taste vector is the mean of the tracks
    they liked; recommendations are the nearest tracks by cosine similarity,
    optionally reinforced by a co-listen (collaborative) signal."""

    def __init__(self, top_k=10, alpha=0.8):
        self.top_k = top_k
        self.alpha = alpha          # content weight; (1-alpha) is collaborative

    def fit(self, features, likes=None):
        X = np.asarray(features, dtype=float)
        # Standardize columns so tempo/energy/embedding share one scale...
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        # ...then L2-normalize rows so a dot product equals cosine similarity.
        self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        # Collaborative co-listen matrix: how often two tracks are liked together.
        self.C = None
        if likes is not None:
            L = np.asarray(likes, dtype=float)      # users x tracks (binary)
            C = L.T @ L                             # track x track co-likes
            np.fill_diagonal(C, 0.0)
            self.C = C
        return self

    def taste(self, liked):
        # User taste = mean of liked tracks' feature vectors, re-normalized.
        v = self.X[liked].mean(0)
        return v / (np.linalg.norm(v) + 1e-12)

    def recommend(self, liked, top_k=None):
        k = top_k or self.top_k
        liked = np.asarray(liked)
        content = self.X @ self.taste(liked)        # cosine sim to taste vector
        score = _unit(content)
        if self.C is not None:                      # blend in the collaborative signal
            collab = self.C[:, liked].mean(1)       # avg co-likes with liked tracks
            score = self.alpha * score + (1 - self.alpha) * _unit(collab)
        score[liked] = -np.inf                      # never recommend a liked track
        return np.argsort(score)[::-1][:k]


def _unit(x):
    # Min-max scale a score vector to [0, 1] so content and collab blend fairly.
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


if __name__ == "__main__":
    np.random.seed(0)

    # Planted genre clusters: each genre has a signature audio profile
    # (tempo/energy/danceability) plus its own embedding vector. Tracks are
    # samples around their genre's prototype, so same-genre tracks cluster
    # in feature space while individual tracks stay noisy.
    n_genres, per_genre, embed_dim = 6, 60, 4
    audio_proto = np.random.randn(n_genres, 3) * 1.5     # tempo, energy, dance
    embed_proto = np.random.randn(n_genres, embed_dim) * 1.5

    feats, genre = [], []
    for g in range(n_genres):
        for _ in range(per_genre):
            audio = audio_proto[g] + 0.6 * np.random.randn(3)
            embed = embed_proto[g] + 0.6 * np.random.randn(embed_dim)
            feats.append(np.concatenate([audio, embed]))
            genre.append(g)
    feats = np.array(feats)
    genre = np.array(genre)
    N = len(genre)

    # Users: each has one preferred genre. Their "likes" are mostly tracks from
    # that genre (with a little cross-genre noise), forming the collaborative data.
    n_users, n_liked = 200, 12
    pref = np.random.randint(n_genres, size=n_users)
    likes = np.zeros((n_users, N))
    liked_idx = []
    for u in range(n_users):
        pool = np.where(genre == pref[u])[0]
        chosen = np.random.choice(pool, size=n_liked, replace=False)
        if np.random.rand() < 0.5:                       # sprinkle 1 off-genre like
            chosen = np.append(chosen, np.random.randint(N))
        likes[u, chosen] = 1.0
        liked_idx.append(chosen)

    model = MusicRecommender(top_k=10, alpha=0.8).fit(feats, likes)

    # Metric: of each user's top-K recommendations (unseen tracks), what fraction
    # match their planted preferred genre? Averaged over all users.
    K = 10
    hits = 0
    for u in range(n_users):
        rec = model.recommend(liked_idx[u], top_k=K)
        hits += np.sum(genre[rec] == pref[u])
    precision = hits / (n_users * K)
    rand = 1.0 / n_genres                                # a random track's genre-match rate

    # One concrete example.
    u0 = 0
    rec0 = model.recommend(liked_idx[u0], top_k=K)
    print("Tracks x features:   %d x %d  (%d genres)" % (N, feats.shape[1], n_genres))
    print("User %d prefers genre %d" % (u0, pref[u0]))
    print("Top-%d recommended genres: %s" % (K, genre[rec0].tolist()))
    print("-" * 60)
    print("Genre Precision@%d:    %.3f" % (K, precision))
    print("Random baseline:      %.3f" % rand)
    print("Improvement:          %.1fx" % (precision / rand))
    print("Beats random:         %s" % bool(precision > 3 * rand))
