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
        steps = 20
        points = [[225 + 25*cos(2*pi*t/steps), 50 + 25*sin(2*pi*t/steps)] for t in range(steps)]
        self.obstacles.append(points)

        # Middle ellipse
        points = [[150 + 40*cos(2*pi*t/steps), 100 + 20*sin(2*pi*t/steps)] for t in range(steps)]
        self.obstacles.append(points)

        # Create visualization
        obstacle_image = np.array([[int(self.is_colliding((row, col))) for col in range(self.width)]
                                   for row in range(self.height)], dtype=np.uint8)
        self.show(obstacle_image)

    def is_colliding(self, point):
        """
        This function calculates the relative angle between the point and all the vertices of the image (between
        -180 and 180 degrees).
        If the sign of this direction ever changes, the point lies outside of the polygon.  Otherwise, the point
        must lie outside of the polygon.
        :param point: The point to check for collisions
        :return: True if there is a collision, False otherwise
        """
        ry, rx = point
        for i in range(len(self.obstacles)):
            direction = 0.0
            collision = True
            for j in range(len(self.obstacles[i]) + 1):
                vx, vy = self.obstacles[i][(j+1) % len(self.obstacles[i])]
                new_direction = atan2(vy - ry, vx - rx)
                new_direction = new_direction if new_direction >= 0.0 else (new_direction + 2.0 * pi)
                difference = new_direction - direction
                difference = difference if difference >= -pi else difference + 2.0 * pi
                difference = difference if difference <= pi else difference - 2.0 * pi
                # print(self.obstacles[i][j], [rx, ry], int(direction*180/pi), int(new_direction*180/pi), int(difference*180/pi))
                direction = new_direction
                if j > 0 > difference:
                    collision = False
            if collision:
                return True
        # print("-----------------------------------------")
        return False


if __name__ == '__main__':
    ObstacleMap(2)
