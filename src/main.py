import sys
import numpy as np
import random
import os
from pathlib import Path

POP_SIZE = 100
MAX_DEPTH = 5
N_GENERATIONS = 100
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
ELITISM = 1
BLOAT_PENALTY = 0.01
PARTIAL_REINIT_EVERY = 10
PARTIAL_REINIT_RATIO = 0.2
UNARY_OPERATORS = ['neg', 'sin', 'cos', 'exp', 'sqrt']
BINARY_OPERATORS = ['add', 'sub', 'mul', 'div']

def random_variable(n_features):
    i = random.randint(0, n_features - 1)
    return ('x', i)

def random_constant():
    c = np.round(random.uniform(-1, 1), 3)
    return ('const', c)

def is_variable(value):
    return isinstance(value, tuple) and value[0] == 'x'

def is_constant(value):
    return isinstance(value, tuple) and value[0] == 'const'

class Node:
    def __init__(self, op=None, value=None, children=None):
        self.op = op
        self.value = value
        if children is None:
            self.children = []
        else:
            self.children = children

    def __str__(self):
        if self.op is None:
            if is_variable(self.value):
                return f"x[{self.value[1]}]"
            else:
                return str(self.value[1])
        elif len(self.children) == 1:
            return f"{self.op}({self.children[0]})"
        elif len(self.children) == 2:
            return f"({self.children[0]} {self.op} {self.children[1]})"
        return "N/A"

def tree_size(node):
    count = 1
    for child in node.children:
        count += tree_size(child)
    return count

def tree_depth(node):
    if len(node.children) == 0:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)

def generate_random_tree(max_depth, n_features, grow=True):
    if max_depth == 0:
        if random.random() < 0.5:
            return Node(op=None, value=random_variable(n_features))
        else:
            return Node(op=None, value=random_constant())
    else:
        if not grow:
            node_type = random.choice(['unary', 'binary'])
        else:
            node_type = random.choice(['unary', 'binary', 'leaf'])

        if node_type == 'leaf' and grow:
            if random.random() < 0.5:
                return Node(op=None, value=random_variable(n_features))
            else:
                return Node(op=None, value=random_constant())

        elif node_type == 'unary':
            op = random.choice(UNARY_OPERATORS)
            child = generate_random_tree(max_depth - 1, n_features, grow)
            return Node(op=op, children=[child])
        else:
            op = random.choice(BINARY_OPERATORS)
            left_child = generate_random_tree(max_depth - 1, n_features, grow)
            right_child = generate_random_tree(max_depth - 1, n_features, grow)
            return Node(op=op, children=[left_child, right_child])

def evaluate_tree(node, x):
    if node.op is None:
        if is_variable(node.value):
            i = node.value[1]
            return x[i, :]
        else:
            c = node.value[1]
            return np.full(x.shape[1], c, dtype=float)
    else:
        if node.op == 'neg':
            return -evaluate_tree(node.children[0], x)
        elif node.op == 'sin':
            return np.sin(evaluate_tree(node.children[0], x))
        elif node.op == 'cos':
            return np.cos(evaluate_tree(node.children[0], x))
        elif node.op == 'exp':
            child_val = evaluate_tree(node.children[0], x)
            child_val = np.clip(child_val, -20, 20)
            return np.exp(child_val)
        elif node.op == 'sqrt':
            return np.sqrt(np.abs(evaluate_tree(node.children[0], x)))
        elif node.op == 'add':
            return evaluate_tree(node.children[0], x) + evaluate_tree(node.children[1], x)
        elif node.op == 'sub':
            return evaluate_tree(node.children[0], x) - evaluate_tree(node.children[1], x)
        elif node.op == 'mul':
            return evaluate_tree(node.children[0], x) * evaluate_tree(node.children[1], x)
        elif node.op == 'div':
            lv = evaluate_tree(node.children[0], x)
            rv = evaluate_tree(node.children[1], x)
            small_eps = 1e-8
            rv_safe = np.where(np.abs(rv) < small_eps, small_eps, rv)
            return lv / rv_safe
        else:
            return np.zeros(x.shape[1])

def calculate_mse(tree, x, y):
    y_pred = evaluate_tree(tree, x)
    return np.mean((y - y_pred) ** 2)

def fitness_function(tree, x, y):
    mse = calculate_mse(tree, x, y)
    size = tree_size(tree)
    return mse + BLOAT_PENALTY * size

def tournament_selection(population, x, y):
    competitors = random.sample(population, TOURNAMENT_SIZE)
    best = None
    best_fitness = float('inf')
    for individual in competitors:
        f = fitness_function(individual, x, y)
        if f < best_fitness:
            best_fitness = f
            best = individual
    return best

def copy_tree(node):
    new_node = Node(op=node.op, value=node.value)
    for child in node.children:
        new_node.children.append(copy_tree(child))
    return new_node

def get_random_node(node):
    all_nodes = []
    def traverse(current, parent):
        all_nodes.append((current, parent))
        for child in current.children:
            traverse(child, current)
    traverse(node, None)
    return random.choice(all_nodes)

def crossover(parent1, parent2):
    child1 = copy_tree(parent1)
    child2 = copy_tree(parent2)
    node1, _ = get_random_node(child1)
    node2, _ = get_random_node(child2)
    node1.op, node2.op = node2.op, node1.op
    node1.value, node2.value = node2.value, node1.value
    node1.children, node2.children = node2.children, node1.children
    if tree_depth(child1) > MAX_DEPTH:
        child1 = generate_random_tree(MAX_DEPTH, 1, grow=True)
    if tree_depth(child2) > MAX_DEPTH:
        child2 = generate_random_tree(MAX_DEPTH, 1, grow=True)
    return child1, child2

def mutate(individual, n_features):
    mutant = copy_tree(individual)
    node, _ = get_random_node(mutant)
    if node.op is None:
        if random.random() < 0.5:
            if is_variable(node.value):
                node.value = random_constant()
            else:
                node.value = random_variable(n_features)
        else:
            if is_constant(node.value):
                node.value = random_constant()
            else:
                node.value = random_variable(n_features)
    else:
        if len(node.children) == 1:
            node.op = random.choice(UNARY_OPERATORS)
        elif len(node.children) == 2:
            node.op = random.choice(BINARY_OPERATORS)
    if tree_depth(mutant) > MAX_DEPTH:
        node.op = None
        node.children = []
        node.value = random_constant() if random.random() < 0.5 else random_variable(n_features)
    return mutant

def evolve_population(population, x, y, n_features, generation):
    ranked_pop = sorted(population, key=lambda ind: fitness_function(ind, x, y))
    new_population = []
    for i in range(ELITISM):
        new_population.append(ranked_pop[i])
    while len(new_population) < POP_SIZE:
        parent1 = tournament_selection(ranked_pop, x, y)
        parent2 = tournament_selection(ranked_pop, x, y)
        if random.random() < CROSSOVER_RATE:
            off1, off2 = crossover(parent1, parent2)
        else:
            off1 = copy_tree(parent1)
            off2 = copy_tree(parent2)
        if random.random() < MUTATION_RATE:
            off1 = mutate(off1, n_features)
        if random.random() < MUTATION_RATE:
            off2 = mutate(off2, n_features)
        new_population.append(off1)
        if len(new_population) < POP_SIZE:
            new_population.append(off2)
    if generation % PARTIAL_REINIT_EVERY == 0 and generation != 0:
        n_reinit = int(PARTIAL_REINIT_RATIO * POP_SIZE)
        for i in range(n_reinit):
            new_population[-(i+1)] = generate_random_tree(MAX_DEPTH, n_features, grow=True)
    return new_population

def count_nodes(tree):
    if tree.op is None:
        return 1
    else:
        return 1 + sum(count_nodes(child) for child in tree.children)

def log_population_complexity(population):
    complexities = [count_nodes(tree) for tree in population]
    print(f"Average population complexity: {np.mean(complexities):.2f}")

def log_population_depth(population):
    depths = [tree_depth(tree) for tree in population]
    print(f"Average depth: {np.mean(depths):.2f}, Maximum depth: {np.max(depths)}")

def tree_to_expression(node):
    if node.op is None:
        if is_variable(node.value):
            return f"x[{node.value[1]}]"
        else:
            return str(node.value[1])
    elif node.op in UNARY_OPERATORS:
        if node.op == 'neg':
            return f"-({tree_to_expression(node.children[0])})"
        else:
            return f"np.{node.op}({tree_to_expression(node.children[0])})"
    elif node.op in BINARY_OPERATORS:
        left = tree_to_expression(node.children[0])
        right = tree_to_expression(node.children[1])
        return f"({left} {map_operator(node.op)} {right})"
    else:
        return "0"

def map_operator(op):
    return {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/'
    }.get(op, '+')

def update_formula_in_file(formula_str, file_path='s333971.py', function_name='f2'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_lines = []
    inside_function = False
    for line in lines:
        if line.strip().startswith(f"def {function_name}"):
            inside_function = True
            new_lines.append(f"def {function_name}(x: np.ndarray) -> np.ndarray:\n")
            new_lines.append(f"    return {formula_str}\n")
            continue
        if inside_function:
            if line.strip() == "" or line.strip().startswith("def "):
                inside_function = False
        if not inside_function:
            new_lines.append(line)
    with open(file_path, 'w') as file:
        file.writelines(new_lines)
    print(f"Formula updated in {file_path} in function {function_name}.")

def run_gp(x, y, problem_id):
    n_features = x.shape[0]
    population = []
    half = POP_SIZE // 2
    for _ in range(half):
        population.append(generate_random_tree(MAX_DEPTH, n_features, grow=False))
    for _ in range(half, POP_SIZE):
        population.append(generate_random_tree(MAX_DEPTH, n_features, grow=True))
    best_individual = None
    best_fitness = float('inf')
    for gen in range(N_GENERATIONS):
        population = evolve_population(population, x, y, n_features, gen)
        log_population_complexity(population)
        log_population_depth(population)
        current_best = min(population, key=lambda ind: fitness_function(ind, x, y))
        current_fitness = fitness_function(current_best, x, y)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_individual = current_best
            print(f"New best fitness: {best_fitness:.4f}")
        print(f"Gen {gen+1}/{N_GENERATIONS} | MSE (train): {current_fitness:.4f}")
    best_expression = tree_to_expression(best_individual)
    print(f"Best formula: {best_expression}")
    update_formula_in_file(
        best_expression,
        file_path='/Users/stefanoroybisignano/Desktop/P_CI/symbolic_regression_2/src/s333971.py',
        function_name=f'f{problem_id}'
    )
    return best_individual

if __name__ == "__main__":
    data_dir = '/Users/stefanoroybisignano/Desktop/P_CI/symbolic_regression_2/data/raw/'
    output_dir = '/Users/stefanoroybisignano/Desktop/P_CI/symbolic_regression_2/output/'
    os.makedirs(output_dir, exist_ok=True)
    data_files = sorted(Path(data_dir).glob('*.npz'))
    for data_file in data_files:
        problem_id = data_file.stem.split('_')[-1]
        print(f"\n=== Loading data for Problem {problem_id} ===")
        data = np.load(data_file)
        x, y = data['x'], data['y']
        print(f"x shape {x.shape}, y shape {y.shape}")
        if x.shape[0] > x.shape[1]:
            x = x.T
        np.random.seed(42)
        random.seed(42)
        best_tree = run_gp(x, y, problem_id)
    final_mse = calculate_mse(best_tree, x, y)
    print("\n=== FINAL RESULT ===")
    print("Best formula found:", best_tree)
    print(f"MSE: {final_mse:.4f}")
