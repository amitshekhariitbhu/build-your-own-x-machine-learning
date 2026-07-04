import numpy as np

# Build a Contact Tracing System from scratch.
# People move around a 2D world split into COMMUNITIES and randomly bump into
# each other. A disease starts from a few seeds and spreads SI-style: the more
# close-contact time a susceptible person shares with an infected one, the more
# likely they catch it. The tracer NEVER sees the infection dynamics -- it only
# sees (a) the anonymised CONTACT LOG (who was near whom, and for how long) and
# (b) a subset of confirmed cases. From that it must RANK the rest of the
# population by infection risk so the highest-risk people get tested first.
# Scoring uses contact-graph diffusion (a Katz / personalised-PageRank style
# power iteration that spreads risk out from the known cases along the contact
# graph, one hop after another with decay). We score it with ranking AUC and
# precision@k of finding the HIDDEN infected, and it must beat random ranking
# and a case-blind "most-social person" degree baseline.

N, T = 80, 50          # people, time steps
G = 4                  # communities
R = 0.8                # contact radius (same-step distance that counts as contact)
SEED_COMMS = (0, 1, 2)  # outbreak starts in 3 of 4 communities; #4 stays a control


def simulate(seed=0):
    # Synthetic mobility + SI epidemic with planted community structure.
    rng = np.random.RandomState(seed)
    centers = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], float)  # community hubs
    grp = rng.randint(0, G, size=N)
    home = centers[grp] + rng.randn(N, 2) * 1.2                  # each person's base
    pos = home.copy()

    contact = np.zeros((N, N))                # accumulated pairwise contact time
    infected = np.zeros(N, dtype=bool)
    pool = np.where(np.isin(grp, SEED_COMMS))[0]                 # seedable people
    infected[rng.choice(pool, 3, replace=False)] = True         # 3 seed cases
    inf_time = np.where(infected, 0, T + 1)                     # step first infected

    for t in range(1, T):
        # Ornstein-Uhlenbeck-ish walk: drift home + jitter, rare long hops mix groups.
        pos += 0.25 * (home - pos) + rng.randn(N, 2) * 0.6
        hop = rng.rand(N) < 0.02
        pos[hop] = centers[rng.randint(0, G, hop.sum())] + rng.randn(hop.sum(), 2) * 1.2

        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        near = (d < R) & ~np.eye(N, dtype=bool)                  # who is in contact now
        contact += near                                         # log contact duration

        # SI transmission: risk grows with number of infected contacts this step.
        n_inf_contacts = near[:, infected].sum(axis=1)
        p_catch = 1.0 - (1.0 - 0.07) ** n_inf_contacts          # per-step infection prob
        new = (~infected) & (rng.rand(N) < p_catch)
        inf_time[new] = t
        infected |= new

    return contact, infected, inf_time, grp


def diffuse(contact, seed_vec, alpha=0.55, n_iter=40):
    # Spread risk from known cases across the contact graph, hop by hop with
    # decay: r = seed + alpha * P @ r, where P is the row-normalised contact
    # matrix. Converges to a Katz / personalised-PageRank exposure score.
    deg = contact.sum(axis=1, keepdims=True)
    P = contact / np.where(deg > 0, deg, 1.0)                    # row-stochastic
    r = seed_vec.astype(float).copy()
    for _ in range(n_iter):
        r = seed_vec + alpha * (P @ r)
    return r


class ContactTracer:
    """Ranks people by infection risk from a contact log + known cases."""

    def __init__(self, alpha=0.55):
        self.alpha = alpha

    def fit(self, contact):
        self.contact = np.asarray(contact, float)
        return self

    def risk(self, known_cases):
        # Graph-diffusion risk seeded from the confirmed cases.
        seed = np.zeros(self.contact.shape[0])
        seed[list(known_cases)] = 1.0
        return diffuse(self.contact, seed, self.alpha)

    def direct_exposure(self, known_cases):
        # 1-hop baseline: raw contact time with confirmed cases only.
        seed = np.zeros(self.contact.shape[0])
        seed[list(known_cases)] = 1.0
        return self.contact @ seed


def auc(scores, pos_mask):
    # Rank-based AUC = P(random positive ranked above random negative).
    pos, neg = scores[pos_mask], scores[~pos_mask]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.argsort(scores))                      # ranks, ties -> avg-ish
    ranks_pos = order[pos_mask].sum()
    u = ranks_pos - len(pos) * (len(pos) - 1) / 2.0
    return u / (len(pos) * len(neg))


if __name__ == "__main__":
    np.random.seed(0)

    contact, infected, inf_time, grp = simulate(seed=0)
    rng = np.random.RandomState(1)

    # Reveal ~45% of the infected as CONFIRMED cases; the rest stay HIDDEN and
    # are what the tracer must surface. Susceptibles are the true negatives.
    inf_idx = np.where(infected)[0]
    known = set(rng.choice(inf_idx, int(0.45 * len(inf_idx)), replace=False))
    candidates = np.array([i for i in range(N) if i not in known])   # who we rank
    hidden = np.array([infected[i] for i in candidates])             # true positives

    tracer = ContactTracer().fit(contact)
    risk = tracer.risk(known)[candidates]
    exposure = tracer.direct_exposure(known)[candidates]
    degree = contact.sum(axis=1)[candidates]                         # case-blind baseline

    auc_diff = auc(risk, hidden)
    auc_exp = auc(exposure, hidden)
    auc_deg = auc(degree, hidden)

    k = int(hidden.sum())                       # test as many people as truly hidden
    top = np.argsort(-risk)[:k]
    prec_at_k = hidden[top].mean()
    base_rate = hidden.mean()                   # precision of random / majority pick

    print("People: %d   Communities: %d   Steps: %d" % (N, G, T))
    print("Infected: %d   Confirmed cases: %d   Hidden infected: %d   Susceptible: %d"
          % (infected.sum(), len(known), k, (~hidden).sum()))
    print("-" * 64)
    print("Ranking AUC  diffusion tracer : %.3f" % auc_diff)
    print("Ranking AUC  1-hop exposure   : %.3f" % auc_exp)
    print("Ranking AUC  degree (blind)   : %.3f" % auc_deg)
    print("Ranking AUC  random baseline  : 0.500")
    print("-" * 64)
    print("Precision@%d  diffusion tracer : %.3f" % (k, prec_at_k))
    print("Precision@%d  random baseline  : %.3f" % (k, base_rate))
    print("-" * 64)
    beats = (auc_diff > 0.5 and auc_diff > auc_deg and prec_at_k > base_rate)
    print("Contact tracer beats random & case-blind baselines:", beats)
