import numpy as np

# Proximal Policy Optimization (PPO) from scratch.
#
# PPO is an on-policy actor-critic method. What makes it "PPO" (vs plain
# REINFORCE) is the CLIPPED surrogate objective: it reuses each batch of
# collected transitions for several gradient epochs, but clips the policy
# ratio r = pi_new/pi_old to [1-eps, 1+eps] so an update can never move the
# policy too far from the data it was collected under. We also use a learned
# value baseline with Generalized Advantage Estimation (GAE) for the advantages.
#
# Planted MDP: a "corridor" of N cells. The agent starts at cell 0 and must
# reach the goal at cell N-1. Action 1 (right) advances, action 0 (left)
# retreats; each step costs -0.02 and reaching the goal pays +1. The optimal
# policy is therefore "always go right", which PPO must recover from scratch.
class Corridor:
    def __init__(self, n=6, max_steps=40):
        self.n, self.max_steps = n, max_steps

    def _obs(self):
        o = np.zeros(self.n); o[self.pos] = 1.0     # one-hot position
        return o

    def reset(self):
        self.pos, self.t = 0, 0
        return self._obs()

    def step(self, a):
        self.pos = min(self.pos + 1, self.n - 1) if a == 1 else max(self.pos - 1, 0)
        self.t += 1
        reached = self.pos == self.n - 1
        trunc = self.t >= self.max_steps and not reached
        reward = 1.0 - 0.02 if reached else -0.02
        return self._obs(), reward, reached, trunc


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def init_mlp(din, dh, dout, scale=0.1):
    return {"W1": np.random.randn(din, dh) * scale, "b1": np.zeros(dh),
            "W2": np.random.randn(dh, dout) * scale, "b2": np.zeros(dout)}


def mlp_forward(p, X):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.tanh(z1)
    out = a1 @ p["W2"] + p["b2"]
    return out, (X, a1)


def mlp_backward(p, cache, grad_out):
    X, a1 = cache
    g = {"W2": a1.T @ grad_out, "b2": grad_out.sum(0)}
    da1 = grad_out @ p["W2"].T
    dz1 = da1 * (1 - a1 ** 2)                        # tanh'(z) = 1 - tanh(z)^2
    g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
    return g


class Adam:
    def __init__(self, params, lr=0.01):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class PPO:
    def __init__(self, dobs, nact, dh=32, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=8, beta=0.01):
        self.policy = init_mlp(dobs, dh, nact)
        self.value = init_mlp(dobs, dh, 1)
        self.opt_p, self.opt_v = Adam(self.policy, 0.01), Adam(self.value, 0.01)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.beta, self.nact = epochs, beta, nact

    def probs(self, X):
        return softmax(mlp_forward(self.policy, X)[0])

    def collect(self, env, n_episodes):
        # Roll out whole episodes under the current policy; compute GAE per episode.
        O, A, LP, RET, ADV = [], [], [], [], []
        for _ in range(n_episodes):
            obs = env.reset(); eo, ea, er, ev, elp = [], [], [], [], []
            while True:
                p = self.probs(obs[None])[0]
                a = int(np.random.choice(self.nact, p=p))
                v = float(mlp_forward(self.value, obs[None])[0].item())
                nobs, r, done, trunc = env.step(a)
                eo.append(obs); ea.append(a); er.append(r); ev.append(v)
                elp.append(np.log(p[a] + 1e-10))
                obs = nobs
                if done or trunc:
                    # Bootstrap only on time-limit truncation, not on true terminal.
                    boot = float(mlp_forward(self.value, obs[None])[0].item()) if trunc else 0.0
                    break
            ev.append(boot)
            adv, gae = np.zeros(len(er)), 0.0
            for t in range(len(er) - 1, -1, -1):
                delta = er[t] + self.gamma * ev[t + 1] - ev[t]
                gae = delta + self.gamma * self.lam * gae
                adv[t] = gae
            ret = adv + np.array(ev[:-1])
            O += eo; A += ea; LP += elp; RET += list(ret); ADV += list(adv)
        adv = np.array(ADV)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalize advantages
        return (np.array(O), np.array(A), np.array(LP), np.array(RET), adv)

    def update(self, O, A, old_lp, ret, adv):
        B = len(A); idx = np.arange(B)
        for _ in range(self.epochs):
            # ---- policy: clipped surrogate + entropy bonus ----
            logits, cache = mlp_forward(self.policy, O)
            p = softmax(logits)
            logp = np.log(p[idx, A] + 1e-10)
            ratio = np.exp(logp - old_lp)
            # Gradient of the ratio flows only where the clip is NOT active.
            clipped = (((adv > 0) & (ratio > 1 + self.clip)) |
                       ((adv < 0) & (ratio < 1 - self.clip)))
            dlogp = -adv * ratio * (~clipped)           # d(-surrogate)/d(logp)
            onehot = np.zeros_like(p); onehot[idx, A] = 1.0
            glogits = dlogp[:, None] * (onehot - p)
            # entropy bonus: maximize H -> subtract beta*H from loss
            H = -(p * np.log(p + 1e-10)).sum(1)
            glogits += self.beta * p * (np.log(p + 1e-10) + H[:, None])
            self.opt_p.step(self.policy, mlp_backward(self.policy, cache, glogits / B))
            # ---- value: MSE regression to returns ----
            val, vcache = mlp_forward(self.value, O)
            gval = (val[:, 0] - ret)[:, None] / B
            self.opt_v.step(self.value, mlp_backward(self.value, vcache, gval))

    def fit(self, env, iters=60, episodes_per_iter=20):
        for _ in range(iters):
            self.update(*self.collect(env, episodes_per_iter))
        return self


def run_policy(env, policy, episodes=200):
    # Average total episode reward under a state->action policy.
    tot = 0.0
    for _ in range(episodes):
        obs, done = env.reset(), False
        while True:
            obs, r, done, trunc = env.step(policy(obs))
            tot += r
            if done or trunc:
                break
    return tot / episodes


if __name__ == "__main__":
    np.random.seed(0)

    env = Corridor(n=6, max_steps=40)
    agent = PPO(dobs=env.n, nact=2).fit(env)

    greedy = lambda o: int(np.argmax(agent.probs(o[None])[0]))
    random_pi = lambda o: np.random.randint(2)

    base = run_policy(env, random_pi)       # random-action baseline
    learned = run_policy(env, greedy)       # exploit the learned policy
    optimal = 1.0 - 0.02 - 0.02 * (env.n - 2)   # +1, then (n-1) steps of -0.02

    # The optimal policy is "always go right": greedy action should be 1 everywhere.
    acts = [greedy(np.eye(env.n)[s]) for s in range(env.n - 1)]

    print("Random policy    -> avg return: %6.3f" % base)
    print("PPO   (greedy)   -> avg return: %6.3f" % learned)
    print("Optimal (hand)   ->     return: %6.3f" % optimal)
    print("Greedy actions per state (want all 1=right):", acts)
    print("Beats random baseline:", learned > base + 0.3)
    print("Recovered optimal policy:", all(a == 1 for a in acts))
