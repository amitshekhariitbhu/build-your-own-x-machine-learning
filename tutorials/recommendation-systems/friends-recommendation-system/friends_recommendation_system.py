import numpy as np


class FriendRecommender:
    """People-you-may-know via graph link prediction (CN / Jaccard / Adamic-Adar)."""

    def __init__(self, method="adamic_adar"):
        self.method = method    # one of: common, jaccard, adamic_adar

    def fit(self, A):
        # A: symmetric 0/1 adjacency of the (training) friendship graph.
        A = A.astype(float)
        np.fill_diagonal(A, 0.0)
        self.A = A
        deg = A.sum(1)                              # number of friends per user
        cn = A @ A                                  # common-neighbor counts

        if self.method == "common":
            S = cn
        elif self.method == "jaccard":
            # |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
            union = deg[:, None] + deg[None, :] - cn
            S = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
        else:
            # Adamic-Adar: rarer shared friends (low degree) weigh more.
            w = np.zeros_like(deg)
            hub = deg > 1
            w[hub] = 1.0 / np.log(deg[hub])
            S = A @ (w[:, None] * A)                # sum_z A_iz*A_zj / log(deg_z)

        # Never recommend existing friends or self.
        S[A > 0] = -np.inf
        np.fill_diagonal(S, -np.inf)
        self.scores = S
        return self

    def predict(self, u, v):
        return self.scores[u, v]

    def recommend(self, u, k=10):
        # Top-k non-friend candidates for user u, best score first.
        return np.argsort(self.scores[u])[::-1][:k]


def make_social_graph(n_users=300, n_comm=6, p_in=0.22, p_out=0.003, seed=0):
    # Stochastic block model: dense links inside communities, sparse across.
    rng = np.random.RandomState(seed)
    comm = rng.randint(0, n_comm, size=n_users)          # planted community id
    same = comm[:, None] == comm[None, :]
    P = np.where(same, p_in, p_out)
    A = (rng.rand(n_users, n_users) < P).astype(float)
    A = np.triu(A, 1)
    A = A + A.T                                           # symmetric, no self-loops
    return A, comm


def precision_at_k(model, A_train, held_pos, k):
    # Avg Precision@k over users who have >=1 hidden friendship to recover.
    hits, total, cand_rate = 0, 0, []
    for u in range(A_train.shape[0]):
        pos = held_pos[u]
        if not pos:
            continue
        rec = model.recommend(u, k)
        hits += sum(v in pos for v in rec)
        total += k
        n_cand = np.isfinite(model.scores[u]).sum()      # scorable non-friends
        cand_rate.append(len(pos) / max(n_cand, 1))
    return hits / total, float(np.mean(cand_rate))       # precision, random level


if __name__ == "__main__":
    np.random.seed(0)

    # Synthetic social graph with planted community (block) structure.
    A_full, comm = make_social_graph(n_users=300, n_comm=6, seed=0)
    n_edges = int(A_full.sum() // 2)

    # Hide 30% of real edges as the held-out test set to recover.
    rng = np.random.RandomState(1)
    ei, ej = np.where(np.triu(A_full, 1) > 0)
    hide = rng.rand(len(ei)) < 0.30
    A_train = A_full.copy()
    held_pos = [set() for _ in range(A_full.shape[0])]
    for i, j, h in zip(ei, ej, hide):
        if h:
            A_train[i, j] = A_train[j, i] = 0.0          # remove edge from graph
            held_pos[i].add(j)
            held_pos[j].add(i)

    K = 10
    print("Users: %d   Communities: %d   Edges: %d   Hidden: %d"
          % (A_full.shape[0], comm.max() + 1, n_edges, int(hide.sum())))
    print("-" * 52)

    rand_level = None
    for method in ("common", "jaccard", "adamic_adar"):
        model = FriendRecommender(method).fit(A_train)
        prec, rand_level = precision_at_k(model, A_train, held_pos, K)
        print("Precision@%d  %-12s %.4f   (%.1fx random)"
              % (K, method, prec, prec / rand_level))

    print("Random baseline P@%d:            %.4f" % (K, rand_level))

    # Show a concrete recommendation and how community-consistent it is.
    aa = FriendRecommender("adamic_adar").fit(A_train)
    recs = aa.recommend(0, K)
    same = np.mean(comm[recs] == comm[0])
    print("-" * 52)
    print("Top-%d PYMK for user 0:  %s" % (K, recs.tolist()))
    print("Share in user 0's community: %.2f (planted signal recovered)" % same)
    print("Link prediction beats random: %s" % (prec > rand_level))
