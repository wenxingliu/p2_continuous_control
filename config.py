MEMORY_BUFFER = int(1e6)
BATCH_SIZE = 128
UPDATE_FREQUENCY_PER_STEP = 5
GAMMA = .99
LR_A = 5e-4
LR_C = 5e-4
TAU = 1e-3
WEIGHT_DECAY = 0

# d4pg specific
Vmax = 10
Vmin = -10
N_ATOMS = 60
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
TRAJECTORY_LENGTH = 5
EPSILON = 0.3