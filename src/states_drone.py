""" State space for Game of Drones."""

import numpy as np
from enum import Enum
import math

from utils_drone import Vector, Position
from world_settings import *


class Actions(Enum):
    """ Class to represent all possible actions."""
    ENGINES_OFF = 0
    RIGHT_ENGINE_ON = 1
    LEFT_ENGINE_ON = 2
    BOTH_ENGINES_ON = 3


class StateDiscrete:
    """Class to discretize the state space for the agent."""

    def __init__(self, dist_to_target, eff_angle, total_velocity):
        self.dist_to_target = dist_to_target
        self.eff_angle = eff_angle
        self.total_velocity = total_velocity
        self.state = []
        self.state.extend([self.dist_to_target, self.eff_angle, self.total_velocity])

        # Declaring number of buckets for each state to be discretized
        self.state_buckets = STATE_VALUE_BUCKETS

        # Declaring bounds for states to be discretized
        self.state_value_bounds = STATE_VALUE_BOUNDS

    def discretize(self):
        """Method converting state space from continuous to discrete."""
        state_discrete = []
        for i in range(len(self.state)):
            if self.state[i] <= self.state_value_bounds[i][0]:
                bucket_index = 0
            elif self.state[i] >= self.state_value_bounds[i][1]:
                bucket_index = self.state_buckets[i] - 1
            else:
                bound_width = self.state_value_bounds[i][1] - self.state_value_bounds[i][0]
                offset = self.state_buckets[i] * self.state_value_bounds[i][0] / bound_width
                scaling = self.state_buckets[i] / bound_width
                bucket_index = int(math.floor(scaling * self.state[i] - offset))

            state_discrete.append(bucket_index)

        return tuple(state_discrete)


class ControllerDisturbance:
    """Class implementing probable disturbances from controller."""

    def __init__(self, action, eps=0.9):
        self.action = action
        self.eps = eps

    def set_disturbance(self):
        number_of_actions = max([a.value for a in Actions])
        if np.random.rand() < self.eps:
            return self.action
        else:
            random_action = np.random.randint(0, number_of_actions + 1)
            return Actions(random_action)


class States:
    """Class to represent the states of the environment."""

    def __init__(self, ground_position: Position, drone_position: Position, drone_body_position: Position,
                 target_position: Position, velocity: Vector, angle, angular_velocity=None):
        # Init the position of the environments
        # Ground position
        self._ground = ground_position

        # Distance of left foot to ground
        self._ground_distance = drone_position - self._ground

        # XY coordinates of left foot
        self._x = drone_position.x
        self._y = drone_position.y

        # XY coordinates of target
        self._target_position = target_position

        # Distance of drone's body to target
        self._distance = self.target_pos - drone_body_position

        # Velocity of the drone
        self._velocity = velocity

        # Init angle information, angle velocity is optional
        self._angle = angle
        self._angle_vel = angular_velocity / FPS

        # Left and right dispersions
        self._dispersion_left = 0
        self._dispersion_right = 0

    def __repr__(self):
        return f'Drone distance to target: {round(self.dist_to_target, 4)}, ' \
               f'angle: {round(self._angle, 4)}, ' \
               f'velocity {round(self.total_velocity, 4)} '

    @property
    def ground_pos(self):
        """Returning ground position."""
        return self._ground

    @property
    def target_pos(self):
        """Returning target position."""
        return self._target_position

    @property
    def x(self):
        """Returning drone's x coordinate."""
        return self._x

    @property
    def y(self):
        """Returning drone's y coordinate."""
        return self._y

    @property
    def distance_vector(self):
        """Returning vector of distance between drone's body and target."""
        return self._distance

    @property
    def drone_position(self):
        """Returning the position of the drone."""
        return Position(self.x, self.y)

    def get_position(self):
        """Returning the position of the drone."""
        return self.drone_position

    @property
    def velocity(self):
        """Returning drone's velocity."""
        return self._velocity.normalize_velocity()

    @velocity.setter
    def velocity(self, vel_vector: Vector):
        """
        Setting a new velocity and scale it accordingly to the environment specifications.
        """
        self._velocity.x = vel_vector.x / FPS
        self._velocity.y = vel_vector.y / FPS

    @property
    def vel_x(self):
        """Returning x coordinate of velocity vector."""
        return self._velocity.x

    @property
    def vel_y(self):
        """Returning y coordinate of velocity vector."""
        return self._velocity.y

    @property
    def angle(self):
        """Returning drone's angle."""
        return self._angle / (2 * math.pi)

    @angle.setter
    def angle(self, value):
        """Setting the drone's angle."""
        value *= -1
        if abs(value) > 2 * math.pi:
            value = value % 2 * math.pi
        if value < 0:
            value += 2 * math.pi

        self._angle = value

    @property
    def angle_vel(self):
        """Returning drone's angular velocity."""
        return self._angle_vel

    @angle_vel.setter
    def angle_vel(self, value):
        """Setting drone's angular velocity."""
        self._angle_vel = value

    @property
    def ground_x(self):
        """Returning distance's x coordinate between drone's left foot and ground."""
        return self._ground_distance.x

    @property
    def ground_y(self):
        """Returning distance's y coordinate between drone's left foot and ground."""
        return self._ground_distance.y

    @property
    def ground_distance_vector(self):
        """Returning vector with distance between drone's left foot and ground."""
        return Vector(x=self.ground_x, y=self.ground_y)

    def update_position(self, new_pos: Position):
        """Updating position of drone's left foot and its distance to ground."""
        self._x = new_pos.x
        self._y = new_pos.y
        self._ground_distance.x = new_pos.x - self.ground_pos.x
        self._ground_distance.y = new_pos.y - self.ground_pos.y

    @property
    def dist_to_target(self):
        """Returning distance's length between drone's body and target."""
        return self.distance_vector.length

    def set_drone_position(self, new_pos: Position):
        """Updating the distance to target vector given new drone's body position."""
        self._distance.x = self.target_pos.x - new_pos.x
        self._distance.y = self.target_pos.y - new_pos.y

    @property
    def total_velocity(self):
        """Returning length of velocity vector."""
        return self.velocity.length

    @property
    def heading(self):
        """Defining the direction of the drone movement."""
        return np.array([math.sin(self.angle * (2 * math.pi)),
                         math.cos(self.angle * (2 * math.pi))])

    def get_heading(self):
        """Returning the drone's movement as numpy array."""
        return self.heading

    @property
    def ground_reached(self):
        """Returning if drone has reached the ground."""
        if self.ground_distance_vector.y < (0.2 / H):
            return True
        return False

    @property
    def out_of_window(self):
        """Returning if drone is out of the window."""
        if self._x > 1.0 or self._x < 0.0 or self._y > 1.0:
            return True
        return False

    @property
    def drone_upside_down(self):
        """Drone is upside down. Angle is btw [O 2pi]"""
        if self.angle < (0.5 * math.pi / (2 * math.pi)) or self.angle > (1.5 * math.pi / (2 * math.pi)):
            return False
        return True

    @property
    def target_reached(self):
        """Returning if drone has reached the target."""
        length_norm_factor = np.sqrt(W ** 2 + H ** 2)
        if abs(self.distance_vector.length) < (1.5 / length_norm_factor):
            return True
        return False

    @property
    def dispersion_left(self):
        """Returning the random influence factor of the left engine."""
        return self._dispersion_left

    @dispersion_left.setter
    def dispersion_left(self, value):
        """Setting the random influence factor of the left engine."""
        self._dispersion_left = value

    @property
    def dispersion_right(self):
        """Returning the random influence factor of the right engine."""
        return self._dispersion_right

    @dispersion_right.setter
    def dispersion_right(self, value):
        """Setting the random influence factor of the left engine."""
        self._dispersion_right = value

    @velocity.setter
    def velocity(self, velocity: Vector):
        """Updating the current velocity given new velocity value."""
        self._velocity = ((VIEWPORT_W / SCALE / 2) / FPS) * velocity

    @property
    def optimal_heading_to_target(self):
        """
        Returning the direction of target --> optimal heading for the drone.
        """
        return self.distance_vector.optimal_heading

    def step_reward(self):
        """Returning reward of the current step."""
        rew = - 10 * self.dist_to_target 
        if self.target_reached:
            rew += 250
        if self.drone_upside_down:
            rew -= 10
        if self.out_of_window:
            rew -= 10
        return rew

    def move_pos(self, pos_move: Vector):
        """Moving the position of the drone according to given input vector."""
        self._distance += pos_move

    def update_state(self, left_foot_pos: Position, drone_body_pos: Position, angle: float, velocity: Vector,
                     angular_velocity: float):
        """Updating the state space."""
        self.update_position(Position(left_foot_pos.x, left_foot_pos.y))
        self.set_drone_position(Position(drone_body_pos.x, drone_body_pos.y))
        self.angle = angle
        self.velocity = velocity
        self.angle_vel = angular_velocity

    @property
    def obs_space(self):
        """Returning the observation space."""
        return np.array([self.distance_vector.x, self.distance_vector.y, self.velocity.x, self.velocity.y,
                         self.angle, self.angle_vel])

    def set_state(self, obs_sample):
        """Updating the current state space based on the dynamic state of drone."""
        distance_vec = Vector(x=obs_sample[0], y=obs_sample[1])
        vel = Vector(x=obs_sample[2], y=obs_sample[3])

        drone_body = self.target_pos - distance_vec
        foot_translation = Position(drone_body.x - 0.9 * (LEG_AWAY / SCALE) - (FOOT_WIDTH / SCALE),
                                    drone_body.y - (DRONE_HEIGHT / (2 * SCALE)) - (2 * FOOT_HEIGHT / SCALE) -
                                    (1.75 * LEG_HEIGHT / SCALE) - (2 * TO_ENGINE_HEIGHT / SCALE) - (
                                                2 * ENGINE_HEIGHT / SCALE))
        left_foot_pos = drone_body + foot_translation

        self.update_state(left_foot_pos=left_foot_pos, drone_body_pos=drone_body, velocity=vel,
                          angle=obs_sample[4], angular_velocity=obs_sample[5])
