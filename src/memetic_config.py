# ==============================================
#               CONFIGURATION                  
# ==============================================

# Population parameters
POP_SIZE = 100               # Increased population size for greater diversity
MAX_DEPTH = 7                # Reduced maximum depth to mitigate bloat
N_GENERATIONS = 100          # Increased number of generations
TOURNAMENT_SIZE = 10         # Higher selective pressure
MUTATION_RATE = 0.6          # Reduced to balance the effect of crossover
CROSSOVER_RATE = 0.4         # Increased to encourage exploration
ELITISM = 10                 # Increased number of elite individuals

# Bloat control parameter
BLOAT_PENALTY = 0            # Increased penalty to favor smaller trees

# Partial Reinitialization parameters
PARTIAL_REINIT_EVERY = 100   # Increased reinitialization frequency (obsolete)
PARTIAL_REINIT_RATIO = 0.25  # Increased reinitialization proportion (obsolete)
DIVERSITY_THRESHOLD = 0.1    # Threshold to trigger reinitialization
REINIT_FRACTION = 0.2        # Fraction of the population to reinitialize

# New options
ENABLE_LOCAL_SEARCH = True   # Enable/disable local search
ADAPTIVE_STRATEGY = True     # Enable/disable adaptive strategies

# Early stopping parameters
MAX_GENERATIONS_NO_IMPROVEMENT = 20  # Maximum generations without improvement
FITNESS_THRESHOLD = 1                # Minimum threshold for best fitness

# Seed for reproducibility
SEED = 42

# Artificial Bee Colony (ABC) settings
ENABLE_ABC = True  # Flag to enable ABC algorithm

# Heap-Based Optimizer (HBO) settings
ENABLE_HBO = True             # Flag to enable the Heap-Based Optimizer
HBO_MAX_ITERATIONS = 5        # Maximum number of iterations for HBO
HBO_NEIGHBOR_COUNT = 100      # Number of neighbors to generate per candidate
HBO_THRESHOLD = 1             # Fitness threshold above which HBO is activated
