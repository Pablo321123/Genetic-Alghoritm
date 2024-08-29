import numpy as np
from sklearn.preprocessing import normalize

max_it = 5
P = {}


class Individual:
    def __init__(self, x, z) -> None:
        self._x = x
        self._z = z
        self._z_norm = 0
        self._prob = 0

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z):
        self._z = z

    @property
    def z_norm(self):
        return self._z_norm

    @z_norm.setter
    def z_norm(self, z):
        self._z_norm = z

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, prob):
        self._prob = prob

    def __str__(self) -> str:
        return f"\nx: {self.x}\nz: {self.z}\nz_norm: {self.z_norm}"


class GeneticModel:

    alpine2 = lambda x: np.prod(np.sqrt(x) * np.sin(x))

    def __init__(self, popsize, dimensions, limits) -> None:
        self.popsize = popsize
        self.n = dimensions
        self.limits = limits
        self.P = None
        self.rank = []
        self.lstProb = []

    def assess_fitness(self, Pi):
        return GeneticModel.alpine2(Pi)

    def normalize(self):
        z_values = [p.z for p in self.P]
        z_min = self.limits[0]
        z_max = self.limits[1]

        for p in self.P:
            p.z_norm = (p.z - z_min) / (z_max - z_min)

    def make_linear_rank(self):
        lstRank = []

        self.P = sorted(self.P, key=lambda ind: ind.z)

        max_p = len(self.P)
        min_p = 1
        n_ind = len(self.P)  # number of individuals

        for i in range(len(self.P)):
            lstRank.append(
                min_p + ((max_p - min_p) * ((n_ind - (i + 1)) / (n_ind - 1)))
            )

        self.rank = lstRank
        somaRank = sum(self.rank)
        self.lstProb = [rank / somaRank for rank in self.rank]

    # p_i mean that is possible send a parent choosed
    def select_with_replacement(self, p_i=None):
        index_parent = -1

        parent = np.random.choice(len(self.P), size=len(self.P), p=self.lstProb)

        if p_i == self.P[parent[0]]:
            return self.select_with_replacement(p_i)

        return self.P[parent[0]]

    def crossover(self, parent_a, parent_b):
        pass

    def mutate(self, children):
        pass

    def inicialize_pop(self):
        lst = []
        self.P = np.random.uniform(
            self.limits[0], self.limits[1], (self.popsize, self.n)
        )

        for i in range(len(self.P)):
            lst.append(Individual(self.P[i], 0))
        self.P = lst

        return self.P

    def start(self, max_it):
        best = None

        it = 0
        while it < max_it:
            print(it)
            it = it + 1

            for p in P:
                pop_fitness = self.assess_fitness(p.x)
                if best == None or pop_fitness > best:
                    best = pop_fitness
                p.z = pop_fitness

            Q = {}  # next generations of individuals
            
            self.normalize()
            self.make_linear_rank()

            for i in range(self.popsize / 2):
                parent_a = self.select_with_replacement()
                parent_b = self.select_with_replacement(parent_a)

                children_a, children_b = self.crossover(parent_a, parent_b)
                Q = Q + (self.mutate(children_a), self.mutate(children_b))
            P = Q
