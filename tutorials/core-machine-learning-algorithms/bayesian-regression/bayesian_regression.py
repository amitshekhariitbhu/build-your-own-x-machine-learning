import numpy as np


class BayesianLinearRegression:
    """Bayesian linear regression with a Gaussian prior over weights.

    Prior:      w ~ N(0, alpha^-1 I)   (alpha = prior precision)
    Likelihood: y ~ N(Phi w, beta^-1)  (beta  = noise precision)
    Posterior:  w ~ N(m_N, S_N)        (conjugate, closed form)
    """

    def __init__(self, alpha=1.0, beta=25.0):
        self.alpha = alpha  # prior precision
        self.beta = beta    # noise precision
        self.m_N = None     # posterior mean
        self.S_N = None     # posterior covariance

    def fit(self, Phi, y):
        # Posterior precision then covariance: S_N = (alpha I + beta Phi^T Phi)^-1
        n_features = Phi.shape[1]
        S_N_inv = self.alpha * np.eye(n_features) + self.beta * Phi.T @ Phi
        self.S_N = np.linalg.inv(S_N_inv)

        # Posterior mean: m_N = beta S_N Phi^T y
        self.m_N = self.beta * self.S_N @ Phi.T @ y
        return self

    def predict(self, Phi):
        # Predictive mean = Phi m_N
        mean = Phi @ self.m_N

        # Predictive variance = 1/beta + phi^T S_N phi (noise + model uncertainty)
        var = 1.0 / self.beta + np.sum((Phi @ self.S_N) * Phi, axis=1)
        return mean, var


if __name__ == "__main__":
    np.random.seed(0)

    # Noisy linear data: y = 2 x + 0.5 with Gaussian noise.
    true_w = np.array([0.5, 2.0])  # [bias, slope]
    noise_std = 0.2
    x = np.random.uniform(-1, 1, size=40)
    Phi = np.stack([np.ones_like(x), x], axis=1)  # design matrix [1, x]
    y = Phi @ true_w + noise_std * np.random.randn(x.size)

    model = BayesianLinearRegression(alpha=1.0, beta=1.0 / noise_std ** 2)
    model.fit(Phi, y)

    # Test points: uncertainty should grow as we move away from the data.
    x_test = np.array([0.0, 1.0, 3.0])
    Phi_test = np.stack([np.ones_like(x_test), x_test], axis=1)
    mean, var = model.predict(Phi_test)
    std = np.sqrt(var)

    print("True weights [bias, slope]:     ", np.round(true_w, 3))
    print("Posterior mean weights:         ", np.round(model.m_N, 3))
    print("Posterior std of weights:       ", np.round(np.sqrt(np.diag(model.S_N)), 3))
    print("Test x:                         ", x_test)
    print("Predictive mean at test x:      ", np.round(mean, 3))
    print("Predictive std at test x:       ", np.round(std, 3))
