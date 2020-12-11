"""    Global Environment setting of our drone world."""
FPS = 50
SCALE = 30.0

# Drone's dimensions
DRONE_POLYGON = [
    (0, 20), (5, 25), (35, 25), (40, 20),
    (40, 15), (30, 0), (10, 0), (0, 15), (0, 20)
]
DRONE_WIDTH, DRONE_HEIGHT = 40, 25
DRONE_OFFSET = (DRONE_WIDTH / (2 * SCALE))

# Dimensions of the legs
LEG_WIDTH, LEG_HEIGHT = 3, 17
LEG_TORQUE = 30
LEG_AWAY = 18

# Dimensions of the feet
FOOT_WIDTH, FOOT_HEIGHT = 5, 1.5
FOOT_TORQUE = 30

# Dimensions of the engines
TO_ENGINE_WIDTH, TO_ENGINE_HEIGHT = 15, 3
ENGINE_WIDTH, ENGINE_HEIGHT = 2, 7
HELIPAD_WIDTH, HELIPAD_HEIGHT = 7, 2
HELIPAD_AWAY = 25

# Rendering window
VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30

# Scaling dimensions of the window
W = (VIEWPORT_W / SCALE)
H = (VIEWPORT_H / SCALE)

# Number of edges of terrain
CHUNKS = 25

# Target's dimensions
TARGET_RADIUS = 20
TARGET_POINT_RADIUS = 2
TARGET_LINE = 5

# Power Level
RIGHT_ENGINE_POWER = 1
LEFT_ENGINE_POWER = 1

# Thresholds Drone velocities
DRONE_THRESHOLD_VELOCITY = 7.0
DRONE_THRESHOLD_ANGLE_VELOCITY = 1.0

# Discretization values
STATE_VALUE_BUCKETS = (20, 10, 5)
STATE_VALUE_BOUNDS = {0: [0, 1], 1: [0, 1], 2: [-1, 1]}

INITIAL_RANDOM = 1000.0
