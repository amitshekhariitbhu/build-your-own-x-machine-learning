import numpy as np

# Develop a Deep Q-Network (DQN) from scratch.
# Unlike tabular Q-learning (one value per state), DQN approximates the
# action-value function Q(s, a) with a NEURAL NETWORK so it can generalize
# across states from CONTINUOUS features. We plant a deterministic grid-world
# MDP but feed the agent only normalized (x, y) coordinates (2 numbers), so a
# lookup table is impossible -- the net must learn the shape of Q. The three
# ingredients that make it "DQN" are all hand-built here: (1) an MLP Q-network
# trained by backprop, (2) an experience-replay buffer to decorrelate samples,
# and (3) a slowly-updated target network to stabilize the Bellman target.


class GridWorld:
    # Start top-left, reach goal bottom-right. Each step costs -1, the goal
    # gives +10 and ends the episode; this drives the shortest path.
    def __init__(self, size=5):
        self.size = size
        self.n_actions = 4  # 0=up, 1=down, 2=left, 3=right
        self.n_features = 2

    def obs(self, r, c):
        # Continuous state: coordinates scaled to [0, 1] (no state index given).
        return np.array([r / (self.size - 1), c / (self.size - 1)])

    def reset(self):
        self.r, self.c = 0, 0
        return self.obs(self.r, self.c)

    def step(self, a):
        if a == 0:   self.r -= 1
        elif a == 1: self.r += 1
        elif a == 2: self.c -= 1
        else:        self.c += 1
        self.r = min(max(self.r, 0), self.size - 1)  # walls: stay in place
        self.c = min(max(self.c, 0), self.size - 1)
        if (self.r, self.c) == (self.size - 1, self.size - 1):
            return self.obs(self.r, self.c), 10.0, True
        return self.obs(self.r, self.c), -1.0, False


class QNetwork:
    # 2-layer MLP: features -> ReLU(hidden) -> Q-value per action. Trained with
    # manual backprop + Adam on the mean-squared Bellman error.
    def __init__(self, n_in, n_hidden, n_out, lr=5e-3):
        self.W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_out) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self.lr = lr
        self.names = ["W1", "b1", "W2", "b2"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.names}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.names}
        self.t = 0

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)          # ReLU
        return self.a1 @ self.W2 + self.b2          # linear Q-values

    def train_step(self, X, actions, targets):
        # Gradient of MSE only flows through the action that was actually taken.
        q = self.forward(X)
        n = len(X)
        dq = np.zeros_like(q)
        rows = np.arange(n)
        dq[rows, actions] = (2.0 / n) * (q[rows, actions] - targets)
        grads = {
            "W2": self.a1.T @ dq,
            "b2": dq.sum(0),
        }
        da1 = dq @ self.W2.T
        dz1 = da1 * (self.z1 > 0)                    # ReLU derivative
        grads["W1"] = self.X.T @ dz1
        grads["b1"] = dz1.sum(0)
        self._adam(grads)
        return float(np.mean((q[rows, actions] - targets) ** 2))

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p in self.names:
            g = grads[p]
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * g * g
            mhat = self.m[p] / (1 - b1 ** self.t)
            vhat = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - self.lr * mhat / (np.sqrt(vhat) + eps))

    def clone_weights_from(self, other):
        for p in self.names:
            setattr(self, p, getattr(other, p).copy())


class DQNAgent:
    def __init__(self, env, hidden=64, gamma=0.95, buffer_size=5000, batch=32):
        self.env = env
        self.gamma = gamma
        self.batch = batch
        self.online = QNetwork(env.n_features, hidden, env.n_actions)
        self.target = QNetwork(env.n_features, hidden, env.n_actions)
        self.target.clone_weights_from(self.online)
        self.buf = []
        self.buffer_size = buffer_size

    def act(self, s, eps):
        if np.random.rand() < eps:                  # epsilon-greedy exploration
            return np.random.randint(self.env.n_actions)
        return int(np.argmax(self.online.forward(s[None])[0]))

    def remember(self, *transition):
        self.buf.append(transition)
        if len(self.buf) > self.buffer_size:
            self.buf.pop(0)

    def replay(self):
        if len(self.buf) < self.batch:
            return
        idx = np.random.randint(len(self.buf), size=self.batch)
        s, a, r, s2, done = zip(*[self.buf[i] for i in idx])
        s, s2 = np.array(s), np.array(s2)
        a, r, done = np.array(a), np.array(r), np.array(done, dtype=float)
        # Bellman target uses the FROZEN target network for the bootstrap.
        q_next = self.target.forward(s2).max(axis=1)
        targets = r + self.gamma * q_next * (1.0 - done)
        self.online.train_step(s, a, targets)

    def fit(self, episodes=300, max_steps=60, target_sync=25):
        for ep in range(episodes):
            eps = max(0.05, 1.0 - ep / (0.7 * episodes))   # anneal exploration
            s, done, steps = self.env.reset(), False, 0
            while not done and steps < max_steps:
                a = self.act(s, eps)
                s2, r, done = self.env.step(a)
                self.remember(s, a, r, s2, done)
                self.replay()
                s, steps = s2, steps + 1
            if ep % target_sync == 0:
                self.target.clone_weights_from(self.online)   # stabilize targets
        return self


def run_policy(env, policy, max_steps=60):
    s, done, total, steps = env.reset(), False, 0.0, 0
    while not done and steps < max_steps:
        s, r, done = env.step(policy(s))
        total, steps = total + r, steps + 1
    return total, steps


if __name__ == "__main__":
    np.random.seed(0)

    env = GridWorld(size=5)
    agent = DQNAgent(env).fit(episodes=300)

    greedy = lambda s: int(np.argmax(agent.online.forward(s[None])[0]))
    random_pi = lambda s: np.random.randint(env.n_actions)

    # Baseline: average return of a random policy over many rollouts.
    base_return = float(np.mean([run_policy(env, random_pi)[0] for _ in range(200)]))
    g_return, g_steps = run_policy(env, greedy)

    # Optimal (hand-checked): Manhattan distance = 8 steps; 7 costs of -1 then +10.
    opt_steps = 2 * (env.size - 1)
    opt_return = 10.0 - (opt_steps - 1)

    print("Random policy   -> avg return: %6.2f" % base_return)
    print("DQN greedy      ->     return: %6.2f  in %2d steps" % (g_return, g_steps))
    print("Optimal (hand)  ->     return: %6.2f  in %2d steps" % (opt_return, opt_steps))
    print("Beats random baseline :", g_return > base_return)
    print("Reached goal optimally:", g_steps == opt_steps and g_return == opt_return)
