import numpy as np


class GeneticAlgorithm:
    """A binary-genome genetic algorithm: tournament selection, uniform crossover,
    bit-flip mutation, and elitism. Maximizes an arbitrary fitness(genome) callable."""

    def __init__(self, genome_len, fitness_fn, pop_size=80, mutation_rate=0.02,
                 tournament_k=3, elite=2, generations=120):
        self.genome_len = genome_len
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elite = elite
        self.generations = generations
        self.best_genome_ = None
        self.best_fitness_ = -np.inf
        self.history_ = []

    def _evaluate(self, pop):
        # Fitness of every genome in the population.
        return np.array([self.fitness_fn(g) for g in pop])

    def _select_parent(self, pop, fit):
        # Tournament selection: pick k contestants, keep the fittest.
        idx = np.random.randint(0, len(pop), size=self.tournament_k)
        return pop[idx[np.argmax(fit[idx])]]

    def _crossover(self, a, b):
        # Uniform crossover: each gene independently from parent a or b.
        mask = np.random.rand(self.genome_len) < 0.5
        return np.where(mask, a, b)

    def _mutate(self, genome):
        # Bit-flip mutation at rate mutation_rate.
        flips = np.random.rand(self.genome_len) < self.mutation_rate
        return np.where(flips, 1 - genome, genome)

    def run(self):
        # Random initial population of binary genomes.
        pop = (np.random.rand(self.pop_size, self.genome_len) < 0.5).astype(int)

        for _ in range(self.generations):
            fit = self._evaluate(pop)

            # Track the best-so-far individual.
            gen_best = np.argmax(fit)
            if fit[gen_best] > self.best_fitness_:
                self.best_fitness_ = fit[gen_best]
                self.best_genome_ = pop[gen_best].copy()
            self.history_.append(self.best_fitness_)

            # Elitism: carry the top individuals forward unchanged.
            order = np.argsort(fit)[::-1]
            new_pop = [pop[i].copy() for i in order[:self.elite]]

            # Fill the rest by selecting parents, crossing over, and mutating.
            while len(new_pop) < self.pop_size:
                p1 = self._select_parent(pop, fit)
                p2 = self._select_parent(pop, fit)
                child = self._mutate(self._crossover(p1, p2))
                new_pop.append(child)

            pop = np.array(new_pop)

        return self.best_genome_, self.best_fitness_


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: a hidden target bit-string. Fitness = number of bits that
    # match the target, so the unique global optimum is the target itself with a
    # known, hand-verifiable best fitness equal to the genome length.
    L = 40
    target = (np.random.rand(L) < 0.5).astype(int)

    def fitness(genome):
        return int(np.sum(genome == target))

    ga = GeneticAlgorithm(genome_len=L, fitness_fn=fitness, generations=120)
    best_genome, best_fit = ga.run()

    # BASELINE: pure random search using the SAME number of fitness evaluations.
    n_evals = ga.pop_size * ga.generations
    rand_best = max(
        int(np.sum((np.random.rand(L) < 0.5).astype(int) == target))
        for _ in range(n_evals)
    )
    # Expected fitness of a single random genome is L/2 (each bit matches w.p. 0.5).
    random_level = L / 2

    print("Genome length (max possible fitness): {}".format(L))
    print("Random single-genome expected fitness: {:.1f}".format(random_level))
    print("Random-search best over {} tries:      {}".format(n_evals, rand_best))
    print("GA best fitness:                       {}".format(best_fit))
    print("GA recovered target exactly:           {}".format(bool(np.array_equal(best_genome, target))))
    print("GA bits correct: {}/{} = {:.1%}".format(best_fit, L, best_fit / L))
    print("Improvement over random expectation:   +{:.1f} bits".format(best_fit - random_level))
    assert best_fit == L, "GA failed to find the planted optimum"
    print("PASS: GA reached the exact global optimum and beat random search.")
