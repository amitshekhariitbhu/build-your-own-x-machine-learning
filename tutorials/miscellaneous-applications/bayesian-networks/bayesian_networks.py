import numpy as np

# Discrete Bayesian Network from scratch.
# A Bayesian network is a DAG over discrete variables where each node stores a
# Conditional Probability Table (CPT) P(node | parents). The joint factorizes as
# P(X1..Xn) = prod_i P(Xi | parents(Xi)). We (1) sample synthetic data from a
# KNOWN "true" network by ancestral sampling, (2) LEARN the CPTs back from data
# via maximum likelihood with Laplace smoothing, and (3) answer probabilistic
# queries with EXACT inference by enumeration. No graph/ML library is used.
#
# Planted structure: the classic Sprinkler network (topological order C,S,R,W):
#   Cloudy -> Sprinkler, Cloudy -> Rain, {Sprinkler, Rain} -> WetGrass.


class BayesianNetwork:
    # `parents`: dict node -> tuple of parent nodes, listed in topological order
    # (every parent appears before its children). `cards`: node -> #states.
    def __init__(self, parents, cards):
        self.parents = parents
        self.cards = cards
        self.nodes = list(parents.keys())          # topological order
        self.cpt = {}                              # node -> P(node | parents)

    def fit(self, data, alpha=1.0):
        # Maximum-likelihood CPTs with Laplace (add-alpha) smoothing so unseen
        # parent configurations still yield a valid distribution.
        for node in self.nodes:
            pa = self.parents[node]
            shape = tuple(self.cards[p] for p in pa) + (self.cards[node],)
            counts = np.full(shape, alpha)
            idx = tuple(data[p] for p in pa) + (data[node],)
            np.add.at(counts, idx, 1)              # vectorized count of configs
            self.cpt[node] = counts / counts.sum(axis=-1, keepdims=True)
        return self

    def sample(self, n):
        # Ancestral sampling: draw each node from its CPT given already-drawn
        # parent values. Returns dict node -> int array of length n.
        data = {}
        for node in self.nodes:
            pa = self.parents[node]
            if pa:
                probs = self.cpt[node][tuple(data[p] for p in pa)]  # (n, card)
            else:
                probs = np.broadcast_to(self.cpt[node], (n, self.cards[node]))
            u = np.random.rand(n, 1)
            data[node] = (u < np.cumsum(probs, axis=1)).argmax(axis=1)
        return data

    def _factor(self, node, assignment):
        # P(node = assignment[node] | parents = assignment[parents]).
        key = tuple(assignment[p] for p in self.parents[node]) + (assignment[node],)
        return self.cpt[node][key]

    def _enumerate_all(self, nodes, assignment):
        # Sum-product over the joint; hidden variables are summed out. Works
        # because `nodes` is topological, so a node's parents are set before it.
        if not nodes:
            return 1.0
        y, rest = nodes[0], nodes[1:]
        if y in assignment:
            return self._factor(y, assignment) * self._enumerate_all(rest, assignment)
        total = 0.0
        for v in range(self.cards[y]):
            a = dict(assignment); a[y] = v
            total += self._factor(y, a) * self._enumerate_all(rest, a)
        return total

    def query(self, target, evidence):
        # Exact P(target | evidence) via enumeration-ask, normalized.
        dist = np.zeros(self.cards[target])
        for t in range(self.cards[target]):
            a = dict(evidence); a[target] = t
            dist[t] = self._enumerate_all(self.nodes, a)
        return dist / dist.sum()

    def predict(self, target, evidence_rows):
        # MAP class of `target` for each evidence dict; returns int array.
        return np.array([self.query(target, e).argmax() for e in evidence_rows])


if __name__ == "__main__":
    np.random.seed(0)

    parents = {"C": (), "S": ("C",), "R": ("C",), "W": ("S", "R")}
    cards = {"C": 2, "S": 2, "R": 2, "W": 2}

    # --- Planted "true" network (the structure inference must recover) ---
    true = BayesianNetwork(parents, cards)
    true.cpt = {
        "C": np.array([0.5, 0.5]),
        "S": np.array([[0.5, 0.5], [0.9, 0.1]]),          # P(S | C)
        "R": np.array([[0.8, 0.2], [0.2, 0.8]]),          # P(R | C)
        "W": np.array([[[1.0, 0.0], [0.1, 0.9]],          # P(W | S, R)
                       [[0.1, 0.9], [0.01, 0.99]]]),
    }

    data = true.sample(5000)
    ntr = 4000
    train = {k: v[:ntr] for k, v in data.items()}
    test = {k: v[ntr:] for k, v in data.items()}

    # --- Learn the CPTs from the training samples only ---
    net = BayesianNetwork(parents, cards).fit(train)

    # Task: infer Rain from observed Cloudy, Sprinkler, WetGrass (explaining away).
    ev_rows = [{"C": c, "S": s, "W": w}
               for c, s, w in zip(test["C"], test["S"], test["W"])]
    pred = net.predict("R", ev_rows)
    acc = np.mean(pred == test["R"])
    majority = np.bincount(train["R"], minlength=2).argmax()
    base = np.mean(test["R"] == majority)               # majority-class baseline

    # Exact-inference cross-check: enumerated marginal P(W=1) vs its empirical
    # frequency (an independent hand-verifiable check of the inference engine).
    pw1_inf = net.query("W", {})[1]
    pw1_emp = data["W"].mean()

    print("Bayesian Network: infer Rain | (Cloudy, Sprinkler, WetGrass)")
    print("Learned P(R=1|C=1): %.3f   (planted truth 0.800)" % net.cpt["R"][1, 1])
    print("Majority-class baseline accuracy: %.3f" % base)
    print("Bayesian-network inference accuracy: %.3f" % acc)
    print("Inference beats baseline:", bool(acc > base + 0.15))
    print("Exact P(W=1): enumerated %.3f vs empirical %.3f (match: %s)"
          % (pw1_inf, pw1_emp, bool(abs(pw1_inf - pw1_emp) < 0.02)))
    print("All checks passed:",
          bool(acc > base + 0.15 and abs(pw1_inf - pw1_emp) < 0.02))
