import numpy as np

class GridWorld:
    # Small FrozenLake-like MDP. Agent starts top-left, reaches goal bottom-right.
    # Stepping on a hole ends the episode with no reward; the goal gives +1.
    def __init__(self, rows=4, cols=4, holes=((1, 1), (1, 3), (3, 2)), step_penalty=-0.01):
        self.rows, self.cols = rows, cols
        self.holes = set(holes)
        self.goal = (rows - 1, cols - 1)
        self.start = (0, 0)
        self.step_penalty = step_penalty
        self.n_states = rows * cols
        self.n_actions = 4  # 0 up, 1 down, 2 left, 3 right

    def reset(self):
        self.pos = self.start
        return self._idx(self.pos)

    def _idx(self, pos):
        return pos[0] * self.cols + pos[1]

    def step(self, action):
        # Apply action, clamped to the grid; return (next_state, reward, done).
        r, c = self.pos
        if action == 0:   r = max(r - 1, 0)
        elif action == 1: r = min(r + 1, self.rows - 1)
        elif action == 2: c = max(c - 1, 0)
        elif action == 3: c = min(c + 1, self.cols - 1)
        self.pos = (r, c)
        if self.pos == self.goal:
            return self._idx(self.pos), 1.0, True
        if self.pos in self.holes:
            return self._idx(self.pos), -1.0, True
        return self._idx(self.pos), self.step_penalty, False

class QLearningAgent:
    # Tabular Q-learning with an epsilon-greedy behaviour policy.
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.Q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay

    def act(self, state):
        # Epsilon-greedy: explore at random, otherwise pick the best known action.
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        # Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s_next]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def train(self, env, n_episodes=500, max_steps=100):
        # Run episodes; return the total reward earned in each one.
        rewards = []
        for _ in range(n_episodes):
            s = env.reset()
            total = 0.0
            for _ in range(max_steps):
                a = self.act(s)
                s_next, r, done = env.step(a)
                self.update(s, a, r, s_next, done)
                s, total = s_next, total + r
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards.append(total)
        return rewards

    def predict(self, state):
        # Greedy action from the learned table (no exploration).
        return int(np.argmax(self.Q[state]))

    def run_greedy(self, env, max_steps=100):
        # Follow the greedy policy from the start; return (path, reached_goal).
        s = env.reset()
        path = [s]
        for _ in range(max_steps):
            s, r, done = env.step(self.predict(s))
            path.append(s)
            if done:
                return path, r > 0
        return path, False

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    env = GridWorld()
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.1, gamma=0.95)

    rewards = agent.train(env, n_episodes=500)

    early = np.mean(rewards[:50])   # average reward over the first 50 episodes
    late = np.mean(rewards[-50:])   # average reward over the last 50 episodes
    path, reached = agent.run_greedy(env)

    print("Avg reward (first 50 episodes):", round(early, 3))
    print("Avg reward (last 50 episodes): ", round(late, 3))
    print("Reward improvement:            ", round(late - early, 3))
    print("Greedy policy reaches goal:    ", reached)
    print("Greedy path (state indices):   ", path)
