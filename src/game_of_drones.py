import math
import time

import Box2D
import numpy as np
from Box2D.b2 import (circleShape, fixtureDef, polygonShape, revoluteJointDef, edgeShape)
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding, EzPickle

from PID import PID
from states_drone import States, Actions, ControllerDisturbance, StateDiscrete
from utils_drone import Vector, Position
from world_settings import *


class GameOfDrone(EzPickle):
    """Class implementing the drone's environment/dynamics."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        super().__init__(*args, **kwargs)
        self.seed()
        self.viewer = None

        # Init world
        self.world = Box2D.b2World()

        # Init world' features
        # self.sky = None
        self.terrain = []

        # Init drone's body parts
        self.drone = None
        self.legs = []
        self.feet = []
        self.to_engines = []
        self.engines = []
        self.helipads_outer = []
        self.helipads_inner = []
        self.particles = []
        self.list_to_draw = []

        # Init target position
        self.target_position = None

        # Initial starting position
        self.state_starting_position = Position(0, 0)

        # Init state
        self.state_space = States(Position(), Position(), Position(), Position(), Vector(), 0, 0)
        self.observation_space = spaces.Box(np.array([0, 0, -1, -1, 0, -1]), np.array([1, 1, 1, 1, 1, 1]))
        self.action_space = spaces.Discrete(4)

        # Init controller disturbance
        self.controller_disturbance = False

        # Init wind disturbance, to be fixed accordingly
        self.wind_disturbance = False
        self.wind_settings = None

        self.prev_angle = 0.0

    def seed(self, seed=None):
        """Seeding."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        """Destroying environment's features at every reset."""
        if len(self.terrain) is 0:
            return
        self._clean_terrain(True)
        self._clean_particles(True)
        self._clean_drone(True)

    def _set_wind_settings(self):
        self.wind_settings = np.array([np.random.rand(), np.random.uniform(0.0, 7.5),
                                      np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)])

    def reset(self):
        """Reset the environment."""
        self._destroy()

        # Initialize wind settings
        if self.wind_disturbance:
            self._set_wind_settings()

        # Getting world's xy measurements
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        height = np.random.uniform(H / 3, H / 2.5, size=(CHUNKS + 1,))
        chunk_y = [0.15 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]

        # Setting drone and target starting position
        self.target_position = 0.7 * W, 0.9 * H  # np.random.choice(np.linspace(0.3 * W, 0.7 * W, num=5)),
        # np.random.choice(
        # np.linspace(0.7 * H, 0.9 * H, num=5))
        self.state_starting_position.x = 0.3 * W  # np.random.choice(np.linspace(0.3 * W, 0.7 * W, num=5))

        self.state_starting_position.y = chunk_y[0]
        starting_angle = 0
        for ind, i in enumerate(chunk_x):
            if abs(self.state_starting_position.x - chunk_x[ind]) < 0.5:
                self.state_starting_position.y = chunk_y[ind]
                starting_angle = math.atan((chunk_y[ind + 1] - chunk_y[ind]) / (chunk_x[ind + 1] - chunk_x[ind]))

        # Measurements of the sky
        self.sky_polygon = []
        for i in range(CHUNKS - 1):
            p1_sky = (chunk_x[i], chunk_y[i])
            p2_sky = (chunk_x[i + 1], chunk_y[i + 1])
            self.sky_polygon.append([p1_sky, p2_sky, (p2_sky[0], H), (p1_sky[0], H)])

        # Measurements of the terrain
        terrain_polygon = []
        for i in range(CHUNKS - 1):
            p1_terrain = (chunk_x[i], chunk_y[i])
            p2_terrain = (chunk_x[i + 1], chunk_y[i + 1])
            terrain_polygon.append([(p1_terrain[0], 0), (p2_terrain[0], 0), p2_terrain, p1_terrain])

        # Constructing the terrain as static body
        for i in range(CHUNKS - 1):
            terrain_chunk = self.world.CreateStaticBody(
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x, y) for x, y in terrain_polygon[i]]),
                    density=0,
                    friction=0))
            terrain_chunk.color = (0.4, 0.6, 0.3)
            self.terrain.append(terrain_chunk)

        # Constructing the drone's body
        self.drone = self.world.CreateDynamicBody(
            position=(self.state_starting_position.x - (DRONE_WIDTH / (2 * SCALE)),
                      self.state_starting_position.y + (DRONE_HEIGHT / (2 * SCALE)) + (LEG_HEIGHT / SCALE)),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLYGON]),
                density=10.0,
                friction=0,
                restitution=0,
                categoryBits=0x0010,
                maskBits=0x001))
        self.drone.ground_contact = False
        self.drone.color1 = (0.9, 0.9, 0.9)
        self.drone.color2 = (0.6, 0.6, 0.6)

        # Constructing the drone's legs and their joints
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(self.state_starting_position.x + (i * 0.9 * LEG_AWAY / SCALE),
                          self.state_starting_position.y + (FOOT_HEIGHT / SCALE) + (
                                  LEG_HEIGHT / SCALE)),
                angle=i * 0.3,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_WIDTH / SCALE, LEG_HEIGHT / SCALE)),
                    density=10.0,
                    friction=0,
                    restitution=0.2))
            leg.ground_contact = False
            leg.color1 = (0.9, 0.9, 0.9)
            leg.color2 = (0.6, 0.6, 0.6)
            rj_leg = revoluteJointDef(
                bodyA=self.drone,
                bodyB=leg,
                anchor=(self.state_starting_position.x + (i * 0.9 * LEG_AWAY / SCALE),
                        self.state_starting_position.y + (FOOT_HEIGHT / SCALE) + (
                                LEG_HEIGHT / SCALE)),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_TORQUE,
                motorSpeed=0.3 * i)
            leg.joint = self.world.CreateJoint(rj_leg)
            self.legs.append(leg)

            # Constructing the drone's feet and their joints
            foot = self.world.CreateDynamicBody(
                position=(self.state_starting_position.x + (i * 0.9 * LEG_AWAY / SCALE) + (i * (FOOT_WIDTH / SCALE)),
                          self.state_starting_position.y + (FOOT_HEIGHT / SCALE)),
                angle=i * 0.3,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(FOOT_WIDTH / SCALE, FOOT_HEIGHT / SCALE)),
                    density=10.0,
                    friction=0.2,
                    restitution=0.2,
                    categoryBits=0x0010,
                    maskBits=0x001))
            foot.ground_contact = True
            foot.color1 = (0.3, 0.3, 0.3)
            foot.color2 = (0.2, 0.2, 0.2)
            rj_foot = revoluteJointDef(
                bodyA=leg,
                bodyB=foot,
                anchor=(self.state_starting_position.x + (i * 0.9 * LEG_AWAY / SCALE) + (i * (FOOT_WIDTH / SCALE)),
                        self.state_starting_position.y + (FOOT_HEIGHT / SCALE)),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=FOOT_TORQUE,
                motorSpeed=0.3 * i
            )
            foot.joint = self.world.CreateJoint(rj_foot)
            self.feet.append(foot)

            # Constructing the drone's connections to the engines and their joints
            to_engine = self.world.CreateDynamicBody(
                position=(self.state_starting_position.x + (i * 0.8 * LEG_AWAY / SCALE) +
                          (i * TO_ENGINE_WIDTH / SCALE),
                          self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                          (1.75 * LEG_HEIGHT / SCALE) + (DRONE_HEIGHT / (2 * SCALE))),
                angle=0,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(TO_ENGINE_WIDTH / SCALE, TO_ENGINE_HEIGHT / SCALE)),
                    density=10.0,
                    friction=0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001))
            to_engine.ground_contact = False
            to_engine.color1 = (0.9, 0.9, 0.9)
            to_engine.color2 = (0.6, 0.6, 0.6)
            rj_to_engine = revoluteJointDef(
                bodyA=self.drone,
                bodyB=to_engine,
                anchor=(self.state_starting_position.x + (i * 0.8 * LEG_AWAY / SCALE) + (i * TO_ENGINE_WIDTH / SCALE),
                        self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                        (1.75 * LEG_HEIGHT / SCALE) + (DRONE_HEIGHT / (2 * SCALE))),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=FOOT_TORQUE,
                motorSpeed=0.3 * i)
            to_engine.joint = self.world.CreateJoint(rj_to_engine)
            self.to_engines.append(to_engine)

            # Constructing the drone's engines and their joints
            engine = self.world.CreateDynamicBody(
                position=(self.state_starting_position.x + (i * 0.8 * LEG_AWAY / SCALE) +
                          (i * 2 * TO_ENGINE_WIDTH / SCALE),
                          self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                          (1.75 * LEG_HEIGHT / SCALE) + (2 * TO_ENGINE_HEIGHT / SCALE) + (1.5 * ENGINE_HEIGHT / SCALE)),
                angle=0,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(ENGINE_WIDTH / SCALE, ENGINE_HEIGHT / SCALE)),
                    density=10.0,
                    friction=0.2,
                    restitution=0.2,
                    categoryBits=0x0010,
                    maskBits=0x001))
            engine.ground_contact = False
            engine.color1 = (0.3, 0.3, 0.3)
            engine.color2 = (0.2, 0.2, 0.2)
            rj_engine = revoluteJointDef(
                bodyA=to_engine,
                bodyB=engine,
                anchor=(self.state_starting_position.x + (i * 0.8 * LEG_AWAY / SCALE) +
                        (i * 2 * TO_ENGINE_WIDTH / SCALE),
                        self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                        (1.75 * LEG_HEIGHT / SCALE) + (2 * TO_ENGINE_HEIGHT / SCALE) + (1.5 * ENGINE_HEIGHT / SCALE)),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=FOOT_TORQUE,
                motorSpeed=0.3 * i)
            engine.joint = self.world.CreateJoint(rj_engine)
            self.engines.append(engine)

            # Constructing the drone's outer helipads and their joints
            helipad = self.world.CreateDynamicBody(
                position=(self.state_starting_position.x + (i * 1.5 * LEG_AWAY / SCALE) +
                          (i * 1.55 * TO_ENGINE_WIDTH / SCALE) + (i * 2 * ENGINE_WIDTH / SCALE),
                          self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                          (1.75 * LEG_HEIGHT / SCALE) + (2 * TO_ENGINE_HEIGHT / SCALE) + (2 * ENGINE_HEIGHT / SCALE)),
                angle=0,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(HELIPAD_WIDTH / SCALE, HELIPAD_HEIGHT / SCALE)),
                    density=0.1,
                    friction=0.2,
                    restitution=0.2,
                    categoryBits=0x0010,
                    maskBits=0x001))
            helipad.ground_contact = False
            helipad.color1 = (0.9, 0.9, 0.9)
            helipad.color2 = (0.6, 0.6, 0.6)
            rj_helipad = revoluteJointDef(
                bodyA=engine,
                bodyB=helipad,
                anchor=(self.state_starting_position.x + (i * 1.55 * LEG_AWAY / SCALE) +
                        (i * 1.55 * TO_ENGINE_WIDTH / SCALE) + (i * 2 * ENGINE_WIDTH / SCALE),
                        self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                        (1.75 * LEG_HEIGHT / SCALE) + (2 * TO_ENGINE_HEIGHT / SCALE) + (2 * ENGINE_HEIGHT / SCALE)),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=30,
                motorSpeed=0.5 * i)
            helipad.joint = self.world.CreateJoint(rj_helipad)
            self.helipads_outer.append(helipad)

            # Constructing the drone's inner helipads and their joints
            helipad = self.world.CreateDynamicBody(
                position=(
                    self.state_starting_position.x + (i * 1.1 * LEG_AWAY / SCALE) + (i * TO_ENGINE_WIDTH / SCALE),
                    self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                    (1.75 * LEG_HEIGHT / SCALE) + (2 * TO_ENGINE_HEIGHT / SCALE) + (2 * ENGINE_HEIGHT / SCALE)),
                angle=0,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(HELIPAD_WIDTH / SCALE, HELIPAD_HEIGHT / SCALE)),
                    density=0.1,
                    friction=0.2,
                    restitution=0.2,
                    categoryBits=0x0010,
                    maskBits=0x001))
            helipad.ground_contact = False
            helipad.color1 = (0.9, 0.9, 0.9)
            helipad.color2 = (0.6, 0.6, 0.6)
            rj_helipad = revoluteJointDef(
                bodyA=engine,
                bodyB=helipad,
                anchor=(self.state_starting_position.x + (i * 1.09 * LEG_AWAY / SCALE) + (i * TO_ENGINE_WIDTH / SCALE),
                        self.state_starting_position.y + (2 * FOOT_HEIGHT / SCALE) +
                        (1.75 * LEG_HEIGHT / SCALE) + (2 * TO_ENGINE_HEIGHT / SCALE) + (2 * ENGINE_HEIGHT / SCALE)),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=30,
                motorSpeed=0.3 * i)
            helipad.joint = self.world.CreateJoint(rj_helipad)
            self.helipads_inner.append(helipad)

        self.list_to_draw = [self.drone] + self.legs + self.feet + self.to_engines + \
                            self.engines + self.helipads_outer + self.helipads_inner

        # Initializing the state space and normalizing the states
        self.state_space = States(ground_position=self.state_starting_position.normalise(),
                                  drone_position=self.state_starting_position.set_drone_offset(),
                                  drone_body_position=Position(self.drone.position.x,
                                                               self.drone.position.y).set_drone_body_offset(),
                                  target_position=Position(self.target_position[0],
                                                           self.target_position[1]).normalise(),
                                  velocity=Vector(0, 0),
                                  angle=starting_angle,
                                  angular_velocity=0.0)

        return self.state_space.obs_space

    def step(self, action_value):
        """Returning the next_state, reward and state of the episode."""
        # Setting disturbance in receiving action from controller
        if self.controller_disturbance:
            action_value = ControllerDisturbance(action_value).set_disturbance()

        if isinstance(action_value, int):
            action_value = Actions(action_value)

        # Getting the current heading of the drone
        head = self.state_space.heading

        # Initializing the random dispersion of the engines
        self.state_space.dispersion_left = self.np_random.uniform(0, +1.0) / (SCALE / 200)
        self.state_space.dispersion_right = self.np_random.uniform(0, +1.0) / (SCALE / 200)

        engine_p = 0.0
        engine_power = 0.0

        if action_value == Actions.LEFT_ENGINE_ON:
            engine_p = 0.25
            engine_power = LEFT_ENGINE_POWER
            ox = head[0] * self.state_space.dispersion_left
            oy = head[1] * self.state_space.dispersion_left
            impulse_pos = Position(self.engines[0].position.x, self.engines[0].position.y) + Vector(ox, oy)

        elif action_value == Actions.RIGHT_ENGINE_ON:
            engine_p = 0.25
            engine_power = RIGHT_ENGINE_POWER
            ox = head[0] * self.state_space.dispersion_right
            oy = head[1] * self.state_space.dispersion_right
            impulse_pos = Position(self.engines[1].position.x, self.engines[1].position.y) + Vector(ox, oy)

        elif action_value == Actions.BOTH_ENGINES_ON:
            engine_p = 0.75
            engine_power = LEFT_ENGINE_POWER + RIGHT_ENGINE_POWER
            ox = head[0] * (self.state_space.dispersion_left + self.state_space.dispersion_right)
            oy = head[1] * (self.state_space.dispersion_left + self.state_space.dispersion_right)
            impulse_pos = Position(self.drone.position.x + DRONE_OFFSET, self.drone.position.y) + Vector(ox, oy)
        else:
            ox = 0
            oy = 0
            impulse_pos = Position(self.drone.position.x, self.drone.position.y) + Vector(ox + DRONE_OFFSET, oy)

        # Moving the drone with the set impulse action.
        self.move(impulse_pos, ox, oy, engine_p, engine_power, action_value)

        # Apply drone thresholds
        self._apply_drone_thresholds()

        # Updating the state space with normalized values
        position = self.feet[0].position
        drone_body_position = Position(self.drone.position.x, self.drone.position.y)
        drone_velocity = Vector(self.drone.linearVelocity[0], self.drone.linearVelocity[1])
        self.state_space.update_state(left_foot_pos=Position(position.x, position.y).set_drone_offset(),
                                      drone_body_pos=drone_body_position.set_drone_offset(),
                                      angle=self.drone.angle,
                                      velocity=drone_velocity,
                                      angular_velocity=self.drone.angularVelocity)

        # Calculate the reward
        reward = self.state_space.step_reward()

        # Getting the done flag
        done = self._check_done(action_value)
        return self.state_space.obs_space, reward, done, self.state_space

    def move(self, impulse_pos, ox, oy, engine_power, power_value, action_value):
        """Application of linear impulse and wind disturbances to the drone and particles."""
        p = self._create_particle(0.3, impulse_pos.x, impulse_pos.y, engine_power)
        if action_value == Actions.LEFT_ENGINE_ON:
            self.drone.ApplyLinearImpulse(
                (ox * power_value * engine_power, oy * power_value * engine_power),
                impulse_pos.as_array,
                True)
            p.ApplyLinearImpulse((ox, oy), impulse_pos.as_array, True)
        elif action_value == Actions.RIGHT_ENGINE_ON:
            self.drone.ApplyLinearImpulse(
                (ox * power_value * engine_power, oy * power_value * engine_power),
                impulse_pos.as_array,
                True)
            p.ApplyLinearImpulse((ox, oy), impulse_pos.as_array, True)
        elif action_value == Actions.BOTH_ENGINES_ON:
            self.drone.ApplyLinearImpulse(
                (ox * power_value * engine_power, oy * power_value * engine_power),
                impulse_pos.as_array,
                True)

        # Setting up wind disturbance
        if self.wind_disturbance:
            if self.wind_settings[0] < 0.25:
                # print('Wind blowing from the bottom with: ', self.wind_settings[1] * self.wind_settings[-1])
                wind_force_position = Position(self.drone.position.x, self.drone.position.y) + \
                                      Vector(-0 + DRONE_OFFSET, self.wind_settings[-1])
                self.drone.ApplyForce(force=(0, self.wind_settings[1] * self.wind_settings[-1]),
                                      point=wind_force_position.as_array,
                                      wake=True)
            elif 0.25 <= self.wind_settings[0] < 0.5:
                # print('Wind blowing from the top with: ', -self.wind_settings[1] * self.wind_settings[-1])
                wind_force_position = Position(self.drone.position.x, self.drone.position.y) + \
                                      Vector(-0 + DRONE_OFFSET, self.wind_settings[-1])
                self.drone.ApplyForce(force=(0, -self.wind_settings[1] * self.wind_settings[-1]),
                                      point=wind_force_position.as_array,
                                      wake=True)
            elif 0.5 <= self.wind_settings[0] < 0.75:
                # print('Wind blowing from the left with: ', -self.wind_settings[1] * self.wind_settings[2])
                wind_force_position = Position(self.engines[0].position.x, self.engines[0].position.y) + \
                                      Vector(self.wind_settings[2], 0)
                self.drone.ApplyForce(force=(-self.wind_settings[1] * self.wind_settings[2], 0),
                                      point=wind_force_position.as_array,
                                      wake=True)
            else:
                # print('Wind blowing from the right with: ', self.wind_settings[1] * self.wind_settings[2])
                wind_force_position = Position(self.engines[1].position.x, self.engines[1].position.y) + \
                                      Vector(self.wind_settings[2] + (ENGINE_WIDTH / SCALE), 0)
                self.drone.ApplyForce(force=(self.wind_settings[1] * self.wind_settings[2], 0),
                                      point=wind_force_position.as_array,
                                      wake=True)

        # Simulation of the environment
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    def _apply_drone_thresholds(self):
        curr_vel = Vector(self.drone.linearVelocity[0], self.drone.linearVelocity[1])
        if abs(curr_vel.length) > DRONE_THRESHOLD_VELOCITY:
            scale = curr_vel.length / DRONE_THRESHOLD_VELOCITY
            self.drone.linearVelocity = [curr_vel.x / scale, curr_vel.y / scale]

        if DRONE_THRESHOLD_ANGLE_VELOCITY < self.drone.angularVelocity:
            self.drone.angularVelocity = 1.0
        elif -1 * DRONE_THRESHOLD_ANGLE_VELOCITY > self.drone.angularVelocity:
            self.drone.angularVelocity = -1.0

    def _helipad_ground_distance(self, helipad):
        """Returning distance helipads/ground."""
        return abs(helipad.position.y - self.state_starting_position.y) / H

    def _check_done(self, current_action):
        """
        Check if conditions for continuing the episode met.

        Helipads conditions for precaution.
        """
        if self.state_space.target_reached:
            return True
        elif self.state_space.out_of_window:
            return True
        elif self.state_space.drone_upside_down:
            return True
        elif self._helipad_ground_distance(self.helipads_outer[0]) < (1 / H):
            return True
        elif self._helipad_ground_distance(self.helipads_outer[1]) < (1 / H):
            return True
        return False

    def render(self, mode='rgb_array'):
        """Model and environment visualization."""
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

        # Visualizing the sky
        for p in self.sky_polygon:
            self.viewer.draw_polygon(p, color=(0.9, 0.9, 1.0))

        # Visualizing the terrain
        for obj in self.terrain:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color)

        # Improving target visualization with additional features
        target_transform = rendering.Transform(translation=(self.target_position[0], self.target_position[1]))
        self.viewer.draw_circle((TARGET_RADIUS / SCALE), filled=False,
                                color=(0.6350, 0.0780, 0.1840), linewidth=2).add_attr(target_transform)
        self.viewer.draw_circle((TARGET_POINT_RADIUS / SCALE), filled=False,
                                color=(0.6350, 0.0780, 0.1840), linewidth=2).add_attr(target_transform)

        target_line_1 = (self.target_position[0] - (TARGET_RADIUS / SCALE) - (TARGET_LINE / SCALE),
                         self.target_position[1] - (TARGET_RADIUS / SCALE) - (TARGET_LINE / SCALE))
        target_line_2 = (self.target_position[0] - (TARGET_RADIUS / SCALE) + (TARGET_LINE / SCALE),
                         self.target_position[1] - (TARGET_RADIUS / SCALE) + (TARGET_LINE / SCALE))
        self.viewer.draw_polyline([(target_line_1[0], self.target_position[1]),
                                   (target_line_2[0], self.target_position[1])], color=(0.6350, 0.0780, 0.1840),
                                  linewidth=2)
        self.viewer.draw_polyline([(self.target_position[0], target_line_1[1]),
                                   (self.target_position[0], target_line_2[1])], color=(0.6350, 0.0780, 0.1840),
                                  linewidth=2)
        target_line_3 = (self.target_position[0] + (TARGET_RADIUS / SCALE) - (TARGET_LINE / SCALE),
                         self.target_position[1] + (TARGET_RADIUS / SCALE) - (TARGET_LINE / SCALE))
        target_line_4 = (self.target_position[0] + (TARGET_RADIUS / SCALE) + (TARGET_LINE / SCALE),
                         self.target_position[1] + (TARGET_RADIUS / SCALE) + (TARGET_LINE / SCALE))
        self.viewer.draw_polyline([(target_line_3[0], self.target_position[1]),
                                   (target_line_4[0], self.target_position[1])], color=(0.6350, 0.0780, 0.1840),
                                  linewidth=2)
        self.viewer.draw_polyline([(self.target_position[0], target_line_3[1]),
                                   (self.target_position[0], target_line_4[1])], color=(0.6350, 0.0780, 0.1840),
                                  linewidth=2)

        # Getting color of the particles
        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        # Cleaning particles
        self._clean_particles(False)

        # Visualizing the drone
        for obj in self.particles + self.list_to_draw:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pos_particles = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(pos_particles)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False,
                                            linewidth=2).add_attr(pos_particles)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _create_particle(self, mass, x, y, ttl):
        """Create particles for better visualization of impulse position."""
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,
                restitution=0.3)
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_terrain(self, flag):
        """Clean the terrain's static bodies"""
        while self.terrain and flag:
            self.world.DestroyBody(self.terrain.pop(0))

    def _clean_drone(self, flag):
        """Clean the drone's dynamic bodies."""
        self.world.DestroyBody(self.drone)
        self.drone = None
        while self.legs and flag:
            self.world.DestroyBody(self.legs.pop(0))
        while self.feet and flag:
            self.world.DestroyBody(self.feet.pop(0))
        while self.to_engines and flag:
            self.world.DestroyBody(self.to_engines.pop(0))
        while self.engines and flag:
            self.world.DestroyBody(self.engines.pop(0))
        while self.helipads_outer and flag:
            self.world.DestroyBody(self.helipads_outer.pop(0))
        while self.helipads_inner and flag:
            self.world.DestroyBody(self.helipads_inner.pop(0))

    def _clean_particles(self, flag):
        """Cleaning the particles."""
        while self.particles and (flag or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def close(self):
        """Closing the environment if game over."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env_name = 'GameOfDrones-v0'

    # unwrapped to get rid of this TimeLimitWrapper, which might reset the environment twice and thus breaks ergodicity
    environment = GameOfDrone()  # gym.make(env_name).unwrapped
    # Invoking the initialized state space in init
    states = environment.state_space
    # Setting up PID
    pid = PID(kp=0.5, ki=0.5, kd=0.5)
    current_step = 0
    for _ in range(50):
        # Reset environment
        environment.reset()
        while True:
            # if current_step < 40:
            #     action = Actions.BOTH_ENGINES_ON
            #     # pid.update_targets(states)
            # else:
            #     # action = pid.pid_control(states)
            #     # environment.drone.linearVelocity = [0, environment.drone.linearVelocity[1]]
            #     print("PID controller's chosen action: ", action)
            action = Actions.BOTH_ENGINES_ON
            time.sleep(0.05)
            states, ep_reward, episode_done, _ = environment.step(action)
            environment.render(mode='rgb_array')

            if episode_done:
                environment.close()
                break
            current_step += 1
