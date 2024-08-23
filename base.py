import numpy as np

max_it = 5
P = {}


class GeneticModel:
    def __init__(self, popsize, dimensions, limits) -> None:
        self.popsize = popsize
        self.n = dimensions
        self.limits = limits
        self.P = None

    def assess_fitness(self, Pi):
        pass

    def fitness(self, Pi):
        pass

    def select_with_replacement(self, P):
        pass

    def crossover(self, parent_a, parent_b):
        pass

    def mutate(self, children):
        pass

    def inicialize_pop(self):
        self.P = np.random.uniform(self.limits[0], self.limits[1], (self.popsize, self.n))
        return self.P

    def start(self, max_it):
        best = None

        it = 0
        while it < max_it:
            print(it)
            it = it + 1

            for p in P:
                if best == None or self.fitness(p) > self.fitness(best):
                    best = p
            Q = {}  # next generations of individuals

            for i in range(self.popsize / 2):
                parent_a = self.select_with_replacement(P)
                parent_b = self.select_with_replacement(P)

                children_a, children_b = self.crossover(parent_a, parent_b)
                Q = Q + (self.mutate(children_a), self.mutate(children_b))
            P = Q
