import random
import re

from deap import algorithms, base, creator, tools
import numpy as np
import scipy.stats

np.set_printoptions(precision=2)
random.seed(0)

FAILED_REGEX_SCORE = float('-inf')


class EvolutionaryRegexFeaturizer:

    def __init__(self, regex_comps, init_set, train_x, train_y, num_classes):
        self.regex_comps = regex_comps
        self.init_set = init_set
        self.train_x = train_x
        self.train_y = train_y
        self.num_classes = num_classes

        self.max_valid_char = len(regex_comps) - 1
        self.to_regex = lambda arr: re.compile(''.join(regex_comps[i] for i in arr))
        self.from_regex = lambda s: s.translate({ord(ch): i for i, ch in enumerate(regex_comps)})

    @staticmethod
    def _count_matches(regex, s):
        """Returns the number of times the given regex matches the string."""
        return len(regex.findall(s))

    def _score(self, ind, with_counts=False, smoothing_amount=1):
        """
        Evaluates the fitness of the given individual by counting the total
        number of matches per class (across all training examples) then
        computing the negative entropy (after additive smoothing).
        """
        try:
            regex = self.to_regex(ind)
        except:
            return (FAILED_REGEX_SCORE,)

        match_counts = np.zeros(self.num_classes)
        for x, y in zip(self.train_x, self.train_y):
            match_counts[y] += self._count_matches(regex, x)

        score = -scipy.stats.entropy(match_counts + smoothing_amount)
        if with_counts:
            return score, match_counts
        return (score,)

    def train(self, feature_size, regex_size, num_generations, pop_size,
              cx_prob=0.25, mut_prob=0.1, mut_ind_char_prob=0.1, tourn_size=3):
        """
        Sets up and runs the evolutionary algorithm (EA).
        """
        # Tell DEAP we're maximizing a single objective
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        # Define an individual
        toolbox = base.Toolbox()
        toolbox.register('attr_int', random.choice, self.init_set)
        toolbox.register('individual', tools.initRepeat, creator.Individual,
                         toolbox.attr_int, regex_size)

        # A population is a bunch of individuals
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        pop = toolbox.population(pop_size)

        # Set up EA options
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutUniformInt,
                         low=0, up=self.max_valid_char, indpb=mut_ind_char_prob)
        toolbox.register('select', tools.selTournament, tournsize=tourn_size)
        toolbox.register('evaluate', self._score)

        # Set up Statistics tracker
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('median score', np.median)
        stats.register('best score', np.max)

        # Set up HallOfFame (tracks best individuals over time)
        hof = tools.HallOfFame(feature_size)

        # Run EA
        result_pop, log = algorithms.eaSimple(pop, toolbox, cx_prob, mut_prob,
                                              num_generations, stats, hof, True)
        self.best_regexes = [self.to_regex(ind) for ind in hof]

        # Show training summary
        print('-' * 79)
        for ind in hof:
            pattern = self.to_regex(ind).pattern
            score, counts = self._score(ind, with_counts=True)
            print('{}, {:.3f}, {}'.format(pattern, score, counts))

    def featurize(self, examples):
        """Featurizes the given data matrix (one example per row)."""
        return [[self._count_matches(regex, eg) for regex in self.best_regexes] for eg in examples]
