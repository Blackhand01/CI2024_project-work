# =============================================================================
#                           MEMETIC GP CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# 1) POPULATION PARAMETERS
# -----------------------------------------------------------------------------
POP_SIZE = 100            # Population size (improves diversity)
MAX_DEPTH = 7             # Maximum tree depth (mitigates bloat)
N_GENERATIONS = 100       # Number of generations
TOURNAMENT_SIZE = 10      # Increased selection pressure
MUTATION_RATE = 0.6       # Mutation probability
CROSSOVER_RATE = 0.4      # Crossover probability
ELITISM = 10               # Number of elite individuals

# -----------------------------------------------------------------------------
# 2) BLOAT CONTROL
# -----------------------------------------------------------------------------
BLOAT_PENALTY = 0      # Penalty for large trees

# -----------------------------------------------------------------------------
# 3) REINITIALIZATION PARAMETERS
# -----------------------------------------------------------------------------
DIVERSITY_THRESHOLD = 0.1 # Threshold to trigger reinitialization
REINIT_FRACTION = 0.2     # Fraction of the population to reinitialize

# -----------------------------------------------------------------------------
# 4) LOCAL SEARCH & ADAPTIVE STRATEGY
# -----------------------------------------------------------------------------
ENABLE_LOCAL_SEARCH = True # Enable/disable local search
ADAPTIVE_STRATEGY = True   # Enable/disable adaptive strategies

# -----------------------------------------------------------------------------
# 5) EARLY STOPPING PARAMETERS
# -----------------------------------------------------------------------------
MAX_GENERATIONS_NO_IMPROVEMENT = 20  # Maximum generations without improvement
FITNESS_THRESHOLD = 1                # Minimum fitness threshold for early stopping

# -----------------------------------------------------------------------------
# 6) REPRODUCIBILITY
# -----------------------------------------------------------------------------
SEED = 42                 # Global seed for reproducibility

# -----------------------------------------------------------------------------
# 7) ARTIFICIAL BEE COLONY (ABC) SETTINGS
# -----------------------------------------------------------------------------
ENABLE_ABC = True         # Flag to enable the ABC algorithm
ABC_MAX_TRIALS = 10       # Maximum number of trials before replacing the food source
ABC_NUM_ONLOOKERS = 500   # Number of onlooker bees

# -----------------------------------------------------------------------------
# 8) HEAP-BASED OPTIMIZER (HBO) SETTINGS
# -----------------------------------------------------------------------------
ENABLE_HBO = True         # Flag to enable the Heap-Based Optimizer
HBO_MAX_ITERATIONS = 5    # Maximum iterations for HBO
HBO_NEIGHBOR_COUNT = 100  # Number of neighbors generated per candidate
HBO_THRESHOLD = 1         # Fitness threshold above which to activate HBO
