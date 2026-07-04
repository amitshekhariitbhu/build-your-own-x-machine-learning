import numpy as np

# Policy Gradient (REINFORCE with a baseline) from scratch.
#
# We plant a deterministic grid-world MDP: the agent starts top-left and must
# reach the goal (bottom-right) via the shortest path while avoiding a pit.
# Instead of learning action-values (Q-learning), REINFORCE directly
# parameterizes a stochastic policy pi(a|s) = softmax(theta[s]) and pushes the
# parameters up the gradient of expected return, estimated from sampled
# trajectories:  grad = E[ (G_t - baseline) * grad log pi(a_t|s_t) ].
class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), pit=(2, 2)):
        self.size = size
        self.start = start
        self.goal = goal
        self.pit = pit
        self.n_states = size * size
        self.n_actions = 4  # 0=up, 1=down, 2=left, 3=right

    def _to_s(self, rc):
        return rc[0] * self.size + rc[1]

    def potential(self, s):
        # Phi(s) = -Manhattan distance to goal: dense hint used for reward
        # shaping. Potential-based shaping provably keeps the optimal policy
        # unchanged while making the sparse goal far easier to discover.
        r, c = divmod(s, self.size)
        return -(abs(r - self.goal[0]) + abs(c - self.goal[1]))

    def reset(self):
        return self._to_s(self.start)

    def step(self, s, a):
        r, c = divmod(s, self.size)
        if a == 0:   r -= 1
        elif a == 1: r += 1
        elif a == 2: c -= 1
        else:        c += 1
        # Walls: stepping off the grid keeps you in place
        r = min(max(r, 0), self.size - 1)
        c = min(max(c, 0), self.size - 1)
        rc = (r, c)
        if rc == self.goal:
            return self._to_s(rc), 10.0, True   # reach goal: reward, terminal
        if rc == self.pit:
            return self._to_s(rc), -30.0, True   # fall in pit: penalty, terminal
        return self._to_s(rc), -1.0, False       # step cost drives shortest path


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class REINFORCE:
    def __init__(self, n_states, n_actions, alpha=0.15, gamma=0.98):
        self.theta = np.zeros((n_states, n_actions))  # policy logits per state
        self.alpha = alpha
        self.gamma = gamma

    def probs(self, s):
        return softmax(self.theta[s])

    def act(self, s):
        # Sample an action from the current stochastic policy
        return int(np.random.choice(len(self.theta[s]), p=self.probs(s)))

    def _rollout(self, env, max_steps):
        # Sample one trajectory under the current policy; shape rewards with the
        # potential Phi so the learning signal is dense: F = gamma*Phi(s')-Phi(s)
        s, done, steps = env.reset(), False, 0
        S, A, R = [], [], []
        while not done and steps < max_steps:
            a = self.act(s)
            s2, r, done = env.step(s, a)
            r += self.gamma * env.potential(s2) - env.potential(s)
            S.append(s); A.append(a); R.append(r)
            s, steps = s2, steps + 1
        return S, A, R

    def fit(self, env, episodes=1500, max_steps=60):
        for ep in range(episodes):
            S, A, R = self._rollout(env, max_steps)
            # Discounted return-to-go G_t for each step
            G, g = np.zeros(len(R)), 0.0
            for t in range(len(R) - 1, -1, -1):
                g = R[t] + self.gamma * g
                G[t] = g
            # Baseline = mean return-to-go (variance reduction, keeps grad unbiased)
            adv = G - G.mean()
            # Gradient ascent: grad log softmax = onehot(a) - probs
            for t in range(len(S)):
                p = self.probs(S[t])
                grad = -p
                grad[A[t]] += 1.0
                self.theta[S[t]] += self.alpha * adv[t] * grad
        return self


def run_policy(env, policy, max_steps=100):
    # Roll out a policy(state)->action, return total reward and step count
    s, done, total, steps = env.reset(), False, 0.0, 0
    while not done and steps < max_steps:
        s, r, done = env.step(s, policy(s))
        total, steps = total + r, steps + 1
    return total, steps


if __name__ == "__main__":
    np.random.seed(0)

    env = GridWorld()
    agent = REINFORCE(env.n_states, env.n_actions).fit(env, episodes=3000)

    greedy = lambda s: int(np.argmax(agent.theta[s]))   # exploit learned policy
    random_pi = lambda s: np.random.randint(env.n_actions)

    # Baseline: average return of a random policy over many rollouts
    rand_returns = [run_policy(env, random_pi)[0] for _ in range(300)]
    base_return = float(np.mean(rand_returns))

    g_return, g_steps = run_policy(env, greedy)

    # Optimal path length = Manhattan distance from start to goal (pit avoidable)
    opt_steps = abs(env.goal[0] - env.start[0]) + abs(env.goal[1] - env.start[1])
    opt_return = 10.0 - (opt_steps - 1)  # (opt_steps-1) step costs of -1, then +10

    print("Random policy    -> avg return: %6.2f" % base_return)
    print("Learned (greedy) ->     return: %6.2f  in %d steps" % (g_return, g_steps))
    print("Optimal (hand)   ->     return: %6.2f  in %d steps" % (opt_return, opt_steps))
    print("Beats random baseline:", g_return > base_return)
    print("Reached goal optimally:", g_steps == opt_steps and g_return == opt_return)
