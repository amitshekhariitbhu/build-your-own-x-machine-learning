import numpy as np

# Deterministic grid-world MDP with a planted goal and a trap. The agent
# starts top-left and must reach the goal (bottom-right) via the shortest
# path while avoiding the pit. An actor-critic agent must recover this
# optimal policy purely from scalar reward feedback.
class GridWorld:
    def __init__(self, size=4, start=(0, 0), goal=(3, 3), pit=(1, 2)):
        self.size = size
        self.start, self.goal, self.pit = start, goal, pit
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
            return self._to_s(rc), 10.0, True    # reach goal: reward, terminal
        if rc == self.pit:
            return self._to_s(rc), -10.0, True    # fall in pit: penalty, terminal
        return self._to_s(rc), -1.0, False        # step cost drives shortest path


# One-step actor-critic (Sutton & Barto).
#   Actor:  softmax policy pi(a|s) over preference parameters theta[s, a],
#           trained by the policy gradient  delta * grad log pi(a|s).
#   Critic: state-value V[s] learned online by TD(0). Its TD error
#           delta = r + gamma*V[s'] - V[s] is the advantage signal that
#           tells the actor whether the sampled action beat expectation.
class ActorCritic:
    def __init__(self, n_states, n_actions, alpha_actor=0.1,
                 alpha_critic=0.3, gamma=0.95):
        self.theta = np.zeros((n_states, n_actions))  # actor preferences
        self.V = np.zeros(n_states)                    # critic values
        self.n_actions = n_actions
        self.alpha_actor, self.alpha_critic = alpha_actor, alpha_critic
        self.gamma = gamma

    def policy(self, s):
        # Softmax over action preferences (shifted for numerical stability)
        z = self.theta[s] - np.max(self.theta[s])
        p = np.exp(z)
        return p / p.sum()

    def act(self, s):
        # Sample an action from the stochastic actor
        return int(np.random.choice(self.n_actions, p=self.policy(s)))

    def update(self, s, a, r, s2, done):
        # Critic: TD error is also the advantage estimate for the actor
        target = r + (0.0 if done else self.gamma * self.V[s2])
        delta = target - self.V[s]
        self.V[s] += self.alpha_critic * delta
        # Actor: grad log pi(a|s) for softmax = onehot(a) - pi(.|s)
        grad = -self.policy(s)
        grad[a] += 1.0
        self.theta[s] += self.alpha_actor * delta * grad

    def fit(self, env, episodes=800, max_steps=100):
        for _ in range(episodes):
            s, done, steps = env.reset(), False, 0
            while not done and steps < max_steps:
                a = self.act(s)
                s2, r, done = env.step(s, a)
                self.update(s, a, r, s2, done)
                s, steps = s2, steps + 1
        return self

    def predict(self, s):
        # Greedy action from the learned actor (no exploration)
        return int(np.argmax(self.theta[s]))


def run_policy(env, policy, max_steps=100):
    # Roll out policy(state)->action; return (total_reward, step_count)
    s, done, total, steps = env.reset(), False, 0.0, 0
    while not done and steps < max_steps:
        s, r, done = env.step(s, policy(s))
        total, steps = total + r, steps + 1
    return total, steps


if __name__ == "__main__":
    np.random.seed(0)

    env = GridWorld()
    agent = ActorCritic(env.n_states, env.n_actions).fit(env, episodes=800)

    greedy = lambda s: agent.predict(s)
    random_pi = lambda s: np.random.randint(env.n_actions)

    # Baseline: average return of a random policy over many rollouts
    base_return = float(np.mean([run_policy(env, random_pi)[0] for _ in range(300)]))
    g_return, g_steps = run_policy(env, greedy)

    # Optimal path length = Manhattan distance from start to goal (pit avoidable)
    opt_steps = abs(env.goal[0] - env.start[0]) + abs(env.goal[1] - env.start[1])
    opt_return = 10.0 - (opt_steps - 1)  # (opt_steps-1) step costs of -1, then +10

    print("Random policy   -> avg return: %6.2f" % base_return)
    print("Learned actor   ->     return: %6.2f  in %d steps" % (g_return, g_steps))
    print("Optimal (hand)  ->     return: %6.2f  in %d steps" % (opt_return, opt_steps))
    print("Beats random baseline:", g_return > base_return)
    print("Reached goal optimally:", g_steps == opt_steps and g_return == opt_return)
