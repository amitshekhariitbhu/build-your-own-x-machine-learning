import numpy as np

# Deterministic grid-world MDP with a planted goal and a trap.
# The agent starts top-left and must reach the goal (bottom-right)
# via the shortest path while avoiding the pit. SARSA (on-policy TD
# control) must recover this optimal policy from reward feedback alone.
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
            return self._to_s(rc), -10.0, True   # fall in pit: penalty, terminal
        return self._to_s(rc), -1.0, False       # step cost drives shortest path


class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.5, gamma=0.95):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma

    def act(self, s, eps):
        # Epsilon-greedy exploration
        if np.random.rand() < eps:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2, a2, done):
        # On-policy TD update: bootstrap on the action A2 actually chosen
        # at S2 (not the max), which is what makes this SARSA, not Q-learning.
        target = r + (0.0 if done else self.gamma * self.Q[s2, a2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def fit(self, env, episodes=600, max_steps=100):
        for ep in range(episodes):
            eps = max(0.05, 1.0 - ep / (0.7 * episodes))  # anneal exploration
            s = env.reset()
            a = self.act(s, eps)                          # choose A before loop
            done, steps = False, 0
            while not done and steps < max_steps:
                s2, r, done = env.step(s, a)
                a2 = self.act(s2, eps)                    # choose A' on-policy
                self.update(s, a, r, s2, a2, done)
                s, a, steps = s2, a2, steps + 1           # SARSA: (S,A,R,S',A')
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
    agent = SARSA(env.n_states, env.n_actions).fit(env, episodes=600)

    greedy = lambda s: int(np.argmax(agent.Q[s]))
    random_pi = lambda s: np.random.randint(env.n_actions)

    # Baseline: average return of a random policy over many rollouts
    rand_returns = [run_policy(env, random_pi)[0] for _ in range(200)]
    base_return = float(np.mean(rand_returns))

    g_return, g_steps = run_policy(env, greedy)

    # Optimal path length = Manhattan distance from start to goal (pit avoidable)
    opt_steps = abs(env.goal[0] - env.start[0]) + abs(env.goal[1] - env.start[1])
    opt_return = 10.0 - (opt_steps - 1)  # (opt_steps-1) step costs of -1, then +10

    print("Random policy   -> avg return: %6.2f" % base_return)
    print("Learned greedy  ->     return: %6.2f  in %d steps" % (g_return, g_steps))
    print("Optimal (hand)  ->     return: %6.2f  in %d steps" % (opt_return, opt_steps))
    print("Beats random baseline:", g_return > base_return)
    print("Reached goal optimally:", g_steps == opt_steps and g_return == opt_return)
