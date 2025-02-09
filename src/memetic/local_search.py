import random
import numpy as np
from core.tree import Node
from core.evaluator import Evaluator
from core.safe_math import ALL_OPERATORS

class LocalSearchManager:
    """
    Modular Local Search Manager with optimized computational efficiency.
    Includes hill climbing, tabu search, simulated annealing, random improvement, and optional Glowworm (GSO).
    """

    def __init__(self, statistics, tabu_size=10, initial_temperature=1.0, cooling_rate=0.95):
        self.statistics = statistics
        self.evaluator = Evaluator()
        self.active_strategy = "simulated_annealing"  # Default strategy

        # Tabu Search parameters
        self.tabu_list = set()
        self.tabu_size = tabu_size

        # Simulated Annealing parameters (default values, used locally)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate


    def tweak(self, individual: Node, n_features: int) -> Node:
        candidate = individual.copy_tree()
        node, _ = Node.get_random_node(candidate)
        if node.op is not None:
            valid_ops = [op.name for op in ALL_OPERATORS.values() if op.arity == ALL_OPERATORS[node.op].arity]
            node.op = random.choice(valid_ops)
        else:
            if node.is_variable():
                node.value = ('const', random.uniform(-1, 1))
            else:
                node.value = ('x', random.randint(0, n_features - 1))
        return candidate

    def evaluate_candidates(self, candidates, x, y, bloat_penalty):
        fitnesses = [
            self.evaluator.fitness_function(candidate, x, y, bloat_penalty)
            for candidate in candidates
        ]
        return fitnesses

    def hill_climb(self, individual, x, y, bloat_penalty):
        candidate = self.tweak(individual, x.shape[0])
        if self.evaluator.fitness_function(candidate, x, y, bloat_penalty) < self.evaluator.fitness_function(individual, x, y, bloat_penalty):
            return candidate
        return individual

    def tabu_search(self, individual, x, y, bloat_penalty):
        candidates = [self.tweak(individual, x.shape[0]) for _ in range(5)]
        # Filter out already explored solutions
        candidates = [c for c in candidates if c.tree_to_expression() not in self.tabu_list]
        if not candidates:
            return individual
        fitnesses = self.evaluate_candidates(candidates, x, y, bloat_penalty)
        best_candidate = candidates[np.argmin(fitnesses)]
        self.tabu_list.add(best_candidate.tree_to_expression())
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop()
        return best_candidate

    def simulated_annealing(self, individual, x, y, bloat_penalty):
        # Use a local temperature that resets at each call
        temperature = self.initial_temperature
        candidate = self.tweak(individual, x.shape[0])
        current_fitness = self.evaluator.fitness_function(individual, x, y, bloat_penalty)
        new_fitness = self.evaluator.fitness_function(candidate, x, y, bloat_penalty)
        delta = new_fitness - current_fitness
        # Accept the new solution if it improves or with a certain probability
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            temperature *= self.cooling_rate  # Update local temperature (without saving the new global value)
            return candidate
        return individual

    def random_improvement(self, individual, x, y, bloat_penalty):
        # Generate multiple candidates and return the best improvement
        candidates = [self.tweak(individual, x.shape[0]) for _ in range(5)]
        fitnesses = self.evaluate_candidates(candidates, x, y, bloat_penalty)
        best_candidate = candidates[np.argmin(fitnesses)]
        if min(fitnesses) < self.evaluator.fitness_function(individual, x, y, bloat_penalty):
            return best_candidate
        return individual

    def choose_strategy(self):
        """
        Select the best local search strategy based on current conditions.
        """
        previous_strategy = self.active_strategy
        reason = "Default strategy"

        if self.statistics.generations_no_improvement > 5:
            self.active_strategy = "random_improvement"
            reason = "Stagnation detected: switching to Random Improvement"
        elif self.statistics.generations_no_improvement > 2:
            self.active_strategy = "simulated_annealing"
            reason = "Stagnation detected: switching to Simulated Annealing"

        # if diversity is low, use Tabu Search to avoid revisiting explored solutions
        elif self.statistics.diversity < 0.3:
            self.active_strategy = "tabu_search"
            reason = "Low diversity: switching to Tabu Search"

        # if tree complexity is too high, use Hill Climbing to simplify it
        elif self.statistics.complexity > 6:
            self.active_strategy = "hill_climb"
            reason = "High complexity detected: switching to Hill Climbing"

        # Default strategy: if everything is stable, use Simulated Annealing
        else:
            self.active_strategy = "simulated_annealing"
            reason = "Stable conditions: using Simulated Annealing"

        # If the strategy has changed, update the logger
        if previous_strategy != self.active_strategy:
            self.statistics.update_single_strategy(
                strategy_type="local_search",
                old_strategy=previous_strategy,
                new_strategy=self.active_strategy,
                reason=reason
            )

    def local_search(self, individual, x, y, bloat_penalty):
        self.choose_strategy()
        strategies = {
            "hill_climb": self.hill_climb,
            "tabu_search": self.tabu_search,
            "simulated_annealing": self.simulated_annealing,
            "random_improvement": self.random_improvement,
        }
        return strategies[self.active_strategy](individual, x, y, bloat_penalty)

    def get_active_strategy(self):
        return self.active_strategy
