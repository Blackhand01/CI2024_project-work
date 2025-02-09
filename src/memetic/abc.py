import random
import numpy as np
from core.evaluator import Evaluator
from core.tree import Node, random_variable, random_constant
from core.safe_math import ALL_OPERATORS

class ArtificialBeeColonyManager:
    """
    Simplified implementation of the Artificial Bee Colony (ABC) algorithm (Karaboga 2005).
    Designed to be integrated with the existing GP structure.
    
    - population: list of trees (Node).
    - max_trials: number of attempts before a food source is replaced (scout phase).
    - n_employed: number of employed bees (usually equal to the population size).
    - n_onlookers: number of onlooker bees.
    """
    def __init__(self, statistics, max_trials=5, n_onlookers=50):
        self.statistics = statistics
        self.evaluator = Evaluator()
        self.max_trials = max_trials
        self.n_onlookers = n_onlookers

        # Store how many consecutive times a solution has not improved.
        # Key: index of the individual, Value: counter of unsuccessful attempts.
        self.trial_counters = {}

    def initialize_trial_counters(self, population):
        """Initialize or reset the counters for the entire population."""
        self.trial_counters = {i: 0 for i in range(len(population))}

    def employed_bees_phase(self, population, x, y, bloat_penalty):
        """
        Employed Bees Phase: each individual (food source) generates a nearby variation,
        evaluates its fitness, and if the solution improves, updates it.
        """
        for i, individual in enumerate(population):
            new_solution = self._tweak(individual, x.shape[0])
            current_fitness = self.evaluator.fitness_function(individual, x, y, bloat_penalty)
            new_fitness = self.evaluator.fitness_function(new_solution, x, y, bloat_penalty)

            if new_fitness < current_fitness:
                population[i] = new_solution
                self.trial_counters[i] = 0  # Reset because an improvement was found.
            else:
                self.trial_counters[i] += 1

    def onlooker_bees_phase(self, population, x, y, bloat_penalty):
        """
        Onlooker Bees Phase: selects the best solutions proportionally to their fitness
        and attempts to improve them with small variations.
        """
        fitness_values = np.array([
            self.evaluator.fitness_function(ind, x, y, bloat_penalty) for ind in population
        ])
        
        # Convert fitness to "profitability" by inverting it (lower fitness => higher priority).
        # An offset is used to avoid infinite values.
        offset = 1.0
        profitability = 1.0 / (fitness_values + offset)
        probabilities = profitability / np.sum(profitability)

        # Select n_onlookers food sources based on probabilities.
        for _ in range(self.n_onlookers):
            chosen_idx = np.random.choice(len(population), p=probabilities)
            chosen_individual = population[chosen_idx]

            new_solution = self._tweak(chosen_individual, x.shape[0])
            current_fitness = fitness_values[chosen_idx]
            new_fitness = self.evaluator.fitness_function(new_solution, x, y, bloat_penalty)

            if new_fitness < current_fitness:
                population[chosen_idx] = new_solution
                self.trial_counters[chosen_idx] = 0
                fitness_values[chosen_idx] = new_fitness  # Update the fitness value.

    def scout_bees_phase(self, population, x, y, bloat_penalty):
        """
        Scout Bees Phase: if a solution does not improve for max_trials attempts,
        replace it with a new random solution.
        """
        for i, t_count in self.trial_counters.items():
            if t_count >= self.max_trials:
                # Replace with a new random tree.
                population[i] = self._generate_random_tree(x.shape[0])
                self.trial_counters[i] = 0

    def run_abc(self, population, x, y, bloat_penalty, n_cycles=5):
        """
        Execute the full ABC pipeline for n cycles: employed -> onlooker -> scout.
        """
        self.initialize_trial_counters(population)

        for _ in range(n_cycles):
            self.employed_bees_phase(population, x, y, bloat_penalty)
            self.onlooker_bees_phase(population, x, y, bloat_penalty)
            self.scout_bees_phase(population, x, y, bloat_penalty)

        return population

    # ------------------ Support Methods ------------------
    def _tweak(self, individual: Node, n_features: int) -> Node:
        """
        Generate a 'local' variation of a GP tree.
        This strategy is similar to a small local mutation.
        """
        new_ind = individual.copy_tree()
        node, _ = Node.get_random_node(new_ind)

        if node.op is not None:
            # Change the operator while preserving the same arity.
            current_arity = ALL_OPERATORS[node.op].arity
            valid_ops = [op for op in ALL_OPERATORS.values() if op.arity == current_arity]
            node.op = random.choice(valid_ops).name
        else:
            # For a leaf node: change the variable or constant.
            if node.is_variable():
                node.value = random_constant()
            else:
                node.value = random_variable(n_features)

        return new_ind

    def _generate_random_tree(self, n_features: int):
        """
        Create a new random tree with a small depth,
        to replace "old" solutions.
        """
        return Node.generate_random_tree(max_depth=3, n_features=n_features, grow=True)
