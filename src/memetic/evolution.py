import random
import numpy as np
from core.tree import Node
from memetic.mutation import AdaptiveMutationManager
from memetic.crossover import AdaptiveCrossoverManager
from memetic.selection import AdaptiveSelectionManager
from memetic.local_search import LocalSearchManager
from memetic.abc import ArtificialBeeColonyManager
from memetic.hbo import HeapBasedOptimizer
from core.evaluator import Evaluator

from memetic_config import (
    MAX_DEPTH, ELITISM, POP_SIZE, 
    CROSSOVER_RATE, MUTATION_RATE, ENABLE_LOCAL_SEARCH, DIVERSITY_THRESHOLD,
    REINIT_FRACTION, MAX_GENERATIONS_NO_IMPROVEMENT, FITNESS_THRESHOLD,
    ENABLE_ABC,
    ENABLE_HBO, HBO_MAX_ITERATIONS, HBO_NEIGHBOR_COUNT, HBO_THRESHOLD
)


class GeneticProgramming:
    """
    Class to coordinate the Genetic Programming process with adaptive managers
    and optional local search (memetic approach).
    """
    def __init__(self, n_features, generations, bloat_penalty, stats, progress_bar=None):
        self.n_features = n_features
        self.generations = generations
        self.bloat_penalty = bloat_penalty
        self.stats = stats
        self.progress_bar = progress_bar
        self.evaluator = Evaluator()

        # Initialize adaptive managers
        self.selection_manager = AdaptiveSelectionManager(stats)
        self.crossover_manager = AdaptiveCrossoverManager(stats)
        self.mutation_manager = AdaptiveMutationManager(stats)
        self.local_search_manager = LocalSearchManager(stats)

        # << ABC ADDED: instantiate a manager for the ABC algorithm >>
        self.abc_manager = ArtificialBeeColonyManager(
            statistics=stats,
            max_trials=10,      # Adjustable parameter
            n_onlookers=500     # Adjustable parameter
        )

    def generate_population(self):
        """Create the initial population of trees."""
        return [
            Node.generate_random_tree(MAX_DEPTH, self.n_features, grow=random.random() > 0.5)
            for _ in range(POP_SIZE)
        ]

    def diversity_injection(self, population):
        """
        Inject diversity into the population by reinitializing a fraction of individuals.
        """
        num_to_reinitialize = int(REINIT_FRACTION * POP_SIZE)
        for i in range(num_to_reinitialize):
            population[-(i + 1)] = Node.generate_random_tree(MAX_DEPTH, self.n_features, grow=True)
        return population

    def evolve_population(self, population, generation):
        """
        Evolve the population using adaptive strategies, and optionally apply local search
        to improve individuals.
        """
        # Sort the population by fitness
        ranked_population = sorted(
            population,
            key=lambda ind: self.evaluator.fitness_function(ind, self.x, self.y, self.bloat_penalty)
        )
        new_population = ranked_population[:ELITISM]

        while len(new_population) < POP_SIZE:
            parent1 = self.selection_manager.select(ranked_population, self.x, self.y, self.bloat_penalty)
            parent2 = self.selection_manager.select(ranked_population, self.x, self.y, self.bloat_penalty)

            if random.random() < CROSSOVER_RATE:
                offspring1, offspring2 = self.crossover_manager.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy_tree(), parent2.copy_tree()

            if random.random() < MUTATION_RATE:
                offspring1 = self.mutation_manager.mutate(offspring1, self.n_features)
            if random.random() < MUTATION_RATE:
                offspring2 = self.mutation_manager.mutate(offspring2, self.n_features)

            new_population.append(offspring1)
            if len(new_population) < POP_SIZE:
                new_population.append(offspring2)

        # Inject diversity if the diversity is below a threshold
        if self.stats.diversity < DIVERSITY_THRESHOLD:
            new_population = self.diversity_injection(new_population)

        if ENABLE_LOCAL_SEARCH:
            ls_fraction = 0.2
            num_local_search = max(1, int(len(new_population) * ls_fraction))
            new_population = sorted(
                new_population,
                key=lambda ind: self.evaluator.fitness_function(ind, self.x, self.y, self.bloat_penalty)
            )
            for i in range(num_local_search):
                improved_ind = self.local_search_manager.local_search(
                    new_population[i], self.x, self.y, self.bloat_penalty
                )
                new_population[i] = improved_ind

        return new_population

    def run(self, x, y):
        """
        Execute the Genetic Programming process.
        """
        self.x = x
        self.y = y
        population = self.generate_population()

        for gen in range(self.generations):
            current_best, current_fitness = self.evaluator.get_best_individual(
                population, self.x, self.y, self.bloat_penalty
            )

            # Early stopping criteria
            if (
                current_fitness == 0
                or self.stats.generations_no_improvement >= MAX_GENERATIONS_NO_IMPROVEMENT
                   and current_fitness <= FITNESS_THRESHOLD
            ):
                self.stats.best_fitness = current_fitness
                self.stats.logger.info(
                    f"Early stopping triggered at generation {gen+1}: "
                    f"Best Fitness = {current_fitness:.4f}, "
                    f"Generations Without Improvement = {self.stats.generations_no_improvement}"
                )
                break

            # Save active strategies BEFORE updating statistics
            old_selection = self.selection_manager.get_active_strategy()
            old_crossover = self.crossover_manager.get_active_strategy()
            old_mutation = self.mutation_manager.get_active_strategy()
            old_local_search = self.local_search_manager.get_active_strategy()

            # Update statistics with current generation metrics and active strategies.
            self.stats.update(
                population,
                self.x,
                self.y,
                self.bloat_penalty,
                best_fitness_current=current_fitness,
                active_strategies={
                    "selection": old_selection,
                    "crossover": old_crossover,
                    "mutation": old_mutation,
                    "local_search": old_local_search,
                }
            )

            # Notify any strategy changes.
            self.stats.update_single_strategy(
                "selection", old_selection, self.selection_manager.get_active_strategy()
            )
            self.stats.update_single_strategy(
                "crossover", old_crossover, self.crossover_manager.get_active_strategy()
            )
            self.stats.update_single_strategy(
                "mutation", old_mutation, self.mutation_manager.get_active_strategy()
            )
            self.stats.update_single_strategy(
                "local_search", old_local_search, self.local_search_manager.get_active_strategy()
            )

            # Evolve population (crossover, mutation, local search, etc.)
            population = self.evolve_population(population, gen)

            # << ABC ADDED: execute ABC on the whole population at the end of the generation >>
            if ENABLE_ABC:
                population = self.abc_manager.run_abc(
                    population=population,
                    x=self.x,
                    y=self.y,
                    bloat_penalty=self.bloat_penalty,
                    n_cycles=5  # Number of ABC cycles per generation
                )

            # << HBO ADDED: execute HBO if the current best fitness is above a threshold (too much higher) >>
            if ENABLE_HBO and current_fitness > HBO_THRESHOLD:
                hbo = HeapBasedOptimizer(
                    population=population,
                    x=self.x,
                    y=self.y,
                    bloat_penalty=self.bloat_penalty,
                    max_iterations=HBO_MAX_ITERATIONS,
                    neighbor_count=HBO_NEIGHBOR_COUNT
                )
                best_candidate_hbo, hbo_fitness = hbo.optimize()
                # If HBO improved the fitness, update the best candidate and the population
                if hbo_fitness < current_fitness:
                    population[0] = best_candidate_hbo
                    current_fitness = hbo_fitness

            # Log every 10 generations
            if gen % 10 == 0:
                self.stats.logger.info(
                    f"Generation {gen+1}/{self.generations} - Best Fitness: {current_fitness:.4f}",
                    generation=gen + 1,
                    best_fitness=current_fitness,
                    avg_fitness=np.mean([
                        self.evaluator.fitness_function(ind, self.x, self.y, self.bloat_penalty)
                        for ind in population
                    ]),
                    diversity=self.stats.diversity,
                    complexity=self.stats.complexity,
                    strategies={
                        "selection": self.selection_manager.get_active_strategy(),
                        "crossover": self.crossover_manager.get_active_strategy(),
                        "mutation": self.mutation_manager.get_active_strategy(),
                        "local_search": self.local_search_manager.get_active_strategy(),
                    }
                )
            self.stats.log_current_strategies()

            if self.progress_bar:
                self.progress_bar.update(1)

        return current_best
