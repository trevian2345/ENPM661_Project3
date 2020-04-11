import numpy as np
import cv2
from math import *


class ObstacleMap:
    """
    Class for obstacle map

    Attributes
    ----------
    obstacles: list of np.array
        List of obstacles.
    """
    def __init__(self, distance):
        """
        Initialization of the obstacle map
        Parameters
        ----------
        distance: float
            Minimum distance between the center of the robot from any point in the obstacle space.
            Accounts for both minimum clearance and the radius of the robot.
        """
        self.height = 10.0
        self.width = 10.0
        self.thickness = distance
        self.window_name = "Image"
        self.obstacles = []
        self.create_obstacles()

    def show(self, image):
        cv2.imshow(self.window_name, image * 255)
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)

    def create_obstacles(self):
        """
        Defines obstacles on the map.
        Each set of points is either an array of shape (2, 2) for the top-left and bottom-right corners of rectangles,
        or an array of shape (3) for the (x, y, r) of a circle.
        """
        # Rectangles
        points = np.array([[4.25, 0.25],
                           [5.75, 1.75]], dtype=np.float64)
        self.obstacles.append(points)
        points = np.array([[1.25, 2.25],
                           [2.75, 3.75]], dtype=np.float64)
        self.obstacles.append(points)
        points = np.array([[4.25, 8.25],
                           [5.75, 9.75]], dtype=np.float64)
        self.obstacles.append(points)

        # Circles
        points = np.array([5.0, 5.0, 1.0], dtype=np.float64)
        self.obstacles.append(points)
        points = np.array([2.0, 7.0, 1.0], dtype=np.float64)
        self.obstacles.append(points)
        points = np.array([8.0, 3.0, 1.0], dtype=np.float64)
        self.obstacles.append(points)
        points = np.array([8.0, 7.0, 1.0], dtype=np.float64)
        self.obstacles.append(points)

    def collision_point(self, point):
        """
        This function checks whether a point lies within the obstacle space.

        Returns
        -------
        bool
            True if there is a collision, False otherwise
        """
        py, px = point
        for obs in self.obstacles:
            # Rectangle
            if obs.shape == (2, 2):
                y1, x1 = obs[0]
                y2, x2 = obs[1]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    return True
            # Circle
            else:
                cy, cx, cr = obs
                if sqrt((cx - px) ** 2 + (cy - py) ** 2) <= cr:
                    return True
        return False

    def collision_circle(self, circle):
        """
        This function checks whether a circle lies within the obstacle space.

        Parameters
        ----------
        circle: tuple of float
            The circle to check for, in the form (x, y, r).

        Returns
        -------
        bool
            True if there is a collision, False otherwise.
        """
        cy, cx, cr = circle
        # Check map boundaries
        if not (cr <= cy <= self.height - cr) or not (cr <= cx <= self.width - cr):
            return True

        # Check obstacles
        for obs in self.obstacles:
            # Rectangles
            if obs.shape == (2, 2):
                # Check if the center of the circle lies within the rectangle
                y1, x1 = obs[0]
                y2, x2 = obs[1]
                if y1 <= cy <= y2 and x1 <= cx <= x2:
                    return True

                corners = [(y1, x1), (y1, x2), (y2, x2), (y2, x1)]
                for j in range(len(corners)):
                    # Check if any of the vertices lies within the circle
                    py1, px1 = corners[j]
                    if sqrt((py1 - cy) ** 2 + (px1 - cx) ** 2) < cr:
                        return True

                    # Check if any of the edges of the rectangle intersects the circle
                    py2, px2 = corners[(j + 1) % 4]
                    if j % 2:
                        if abs(px1 - cx) < cr and px1 <= cx <= px2:
                            return True
                    else:
                        if abs(py1 - cy) < cr and py1 <= cy <= py2:
                            return True

            # Circles - check if centers are closer than the sum of the radii
            else:
                py, px, pr = obs
                if sqrt((px - cx) ** 2 + (py - cy) ** 2) < (pr + cr):
                    return True
        return False


if __name__ == '__main__':
    ObstacleMap(2)
