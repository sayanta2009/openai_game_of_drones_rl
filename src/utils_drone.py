import numpy as np
from world_settings import *


class Vector(object):
    """Vector utility."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self._x = x
        self._y = y

    def __repr__(self):
        return "Vector(x=%f, y=%f, length=%f)" % (self.x, self.y, self.length)

    @property
    def x(self):
        """Returning x coordinate of vector."""
        return self._x

    @property
    def y(self):
        """Returning y coordinate of vector."""
        return self._y

    @x.setter
    def x(self, value: float):
        """Setting x coordinate of vector."""
        self._x = value

    @y.setter
    def y(self, value: float):
        """Setting y coordinate of vector."""
        self._y = value

    @property
    def length(self):
        """Returning the total length of the vector."""
        return np.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def as_array(self):
        """Returning vector an numpy array."""
        return np.array([self.x, self.y])

    @property
    def optimal_heading(self):
        """
        Returning the optimal heading based on a distance vector to a target position.

        Numpy handles all the quadrant differences directly.
        The optimal heading is normalized between 0 and 1.
        """
        angle = np.arctan2(self.y, self.x)
        if angle < 0:
            angle += 2 * np.pi
        return angle / (2 * np.pi)

    def scale_vel_x(self, scaling):
        """Scaling x coordinate of velocity."""
        return self.x * scaling

    def scale_vel_y(self, scaling):
        """Scaling y coordinate of velocity."""
        return self.y * scaling

    def normalize_velocity(self):
        """Scaling, setting a threshold and normalizing the velocity."""
        # scaling = ((VIEWPORT_W / SCALE / 2) / FPS)
        # self.x = self.scale_vel_x(scaling)
        # self.y = self.scale_vel_y(scaling)
        return Vector(self.x / DRONE_THRESHOLD_VELOCITY, self.y / DRONE_THRESHOLD_VELOCITY)

    def normalise(self):
        """Returning normalized xy position."""
        return Position(self.x / W, self.y / H)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __iadd__(self, other):
        """Updating the current instance."""
        self._x += other.x
        self._y += other.y
        return self

    def __isub__(self, other):
        """Updating the current instance."""
        self._x -= other.x
        self._y -= other.y
        return self

    def __rmul__(self, other):
        return Vector(other * self.x, other * self.y)


class Position(Vector):
    """Position utility."""

    def __repr__(self):
        return "Position(x=%d, y=%d)" % (self.x, self.y)

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)



    def set_drone_offset(self):
        """Returning position set to drone's left foot."""
        y_offset = (FOOT_HEIGHT / SCALE)
        return Position(self.x, self.y - y_offset).normalise()

    def set_drone_body_offset(self):
        """Returning position set to center of the drone's body."""
        x_offset = ((DRONE_WIDTH / 2) / SCALE)
        y_offset = ((DRONE_HEIGHT / 2) / SCALE)
        return Position(self.x + x_offset, self.y + y_offset).normalise()

