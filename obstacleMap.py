import numpy as np
import cv2
from math import *


class ObstacleMap:
    def __init__(self, distance):
        """
        Initialization of the obstacle map
        :param distance: Minimum distance between the center of the robot from any point in the obstacle space.
                         Accounts for both minimum clearance and the radius of the robot.
        """
        self.height = 200
        self.width = 300
        self.thickness = distance
        self.collisionMatrix = np.zeros((self.height, self.width), dtype=np.uint8)
        self.window_name = "Image"
        self.obstacles = []
        self.obstacle_space = np.array([[]], dtype=np.uint8)
        self.create_obstacles()

    def show(self, image):
        cv2.imshow(self.window_name, image * 255)
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)

    def create_obstacles(self):
        """
        Checks for collisions.
        """
        # Define obstacles in the collision space.
        # Upper-left polygon
        points = [[25, 15],
                  [75, 15],
                  [20, 80]]
        self.obstacles.append(points)
        points = [[75, 15],
                  [100, 50],
                  [75, 80],
                  [50, 50]]
        self.obstacles.append(points)
        points = [[75, 15],
                  [50, 50],
                  [20, 80]]
        self.obstacles.append(points)

        # Lower-left rectangle
        points = [[95, 170],
                  [95 - 75*sqrt(3)/2, 170 - 75/2],
                  [95 - 75*sqrt(3)/2 + 10/2, 170 - 75/2 - 10*sin(60*pi/180)],
                  [95 + 10/2, 170 - 10*sin(60*pi/180)]]
        self.obstacles.append(points)

        # Lower-right rhombus
        points = [[225, 160], [250, 175], [225, 190], [200, 175]]
        self.obstacles.append(points)

        # Upper-right circle
        points = [[225, 50, 25, 25]]
        self.obstacles.append(points)

        # Middle ellipse
        points = [[150, 100, 40, 20]]
        self.obstacles.append(points)

        # Create visualization
        # self.obstacle_space = np.array([[int(self.is_colliding((row, col))) for col in range(self.width)]
        #                                 for row in range(self.height)], dtype=np.uint8)
        # self.show(self.obstacle_space)

    def is_colliding(self, point, thickness=None, check_inside=True):
        """
        This function calculates the relative angle between the point and all the vertices of the image (between
        -180 and 180 degrees).
        If the sign of this direction ever changes, the point lies outside of the polygon.  Otherwise, the point
        must lie outside of the polygon.
        :param check_inside: Whether to check inside the polygon
        :param thickness: distance to use for rigid robot.  Default uses radius and clearance.
        :param point: The point to check for collisions
        :return: True if there is a collision, False otherwise
        """
        t = self.thickness if thickness is None else thickness
        ry = point[0]
        rx = point[1]
        if not (t <= ry <= self.height - t) or not (t <= rx <= self.width - t):
            return True
        for i in range(len(self.obstacles)):
            # Polygons
            if len(self.obstacles[i][0]) == 2:
                direction = 0.0
                collision = True
                for j in range(len(self.obstacles[i]) + 1):
                    # Check if the point is within range of any of the vertices
                    p1 = self.obstacles[i][j % len(self.obstacles[i])]
                    if ((rx - p1[0]) ** 2) + ((ry - p1[1]) ** 2) <= (t ** 2):
                        return True
                    # Check if the point is inside of the polygon
                    elif not check_inside:
                        collision = False
                        continue
                    vx, vy = self.obstacles[i][(j+1) % len(self.obstacles[i])]
                    new_direction = atan2(vy - ry, vx - rx)
                    new_direction = new_direction if new_direction >= 0.0 else (new_direction + 2.0 * pi)
                    difference = new_direction - direction
                    difference = difference if difference >= -pi else difference + 2.0 * pi
                    difference = difference if difference <= pi else difference - 2.0 * pi
                    direction = new_direction
                    if j > 0 > difference:
                        collision = False
                if collision:
                    return True
                else:
                    # Check if the point is within a certain distance of an edge
                    for j in range(len(self.obstacles[i])):
                        x0, y0 = (rx, ry)
                        x1, y1 = self.obstacles[i][j]
                        x2, y2 = self.obstacles[i][(j + 1) % len(self.obstacles[i])]
                        d = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)/sqrt((y2-y1)**2 + (x2-x1)**2)
                        if d < t:
                            dot_product = (x0 - x1) * (x0 - x2) + (y0 - y1) * (y0 - y2)
                            if dot_product < 0:
                                return True
            # Ellipse / circle
            else:
                vx, vy, vw, vh = self.obstacles[i][0]
                vw += t
                vh += t
                if ((vx - rx) ** 2) + (((vy - ry) * vw/vh) ** 2) <= vw ** 2.0:
                    return True
        return False


if __name__ == '__main__':
    ObstacleMap(2)
