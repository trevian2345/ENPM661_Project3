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
        self.baseImage = np.zeros((self.height, self.width), dtype=np.uint8)
        self.obstacleSpace = np.copy(self.baseImage)
        self.window_name = "Image"
        self.half_planes()

    def show(self, image):
        cv2.imshow(self.window_name, image * 255)
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)

    def half_planes(self):
        """
        Generates a map of the obstacle space using half-planes.
        """
        # Define polygons (not circle or ellipse) to be drawn
        # Upper-left polygon
        shapes_to_draw = []
        points = [[25, 185, 75, 185, False],
                  [75, 185, 20, 120, True],
                  [20, 120, 25, 185, False]]
        for i in range(len(points)):
            points[i][1] = self.height - points[i][1]
            points[i][3] = self.height - points[i][3]
        shapes_to_draw.append(points)
        points = [[75, 185, 100, 150, False],
                  [100, 150, 75, 120, True],
                  [75, 120, 50, 150, True],
                  [50, 150, 75, 185, False]]
        for i in range(len(points)):
            points[i][1] = self.height - points[i][1]
            points[i][3] = self.height - points[i][3]
        shapes_to_draw.append(points)
        points = [[75, 185, 50, 150, True],
                  [50, 150, 20, 120, True],
                  [20, 120, 75, 185, False]]
        for i in range(len(points)):
            points[i][1] = self.height - points[i][1]
            points[i][3] = self.height - points[i][3]
        shapes_to_draw.append(points)

        # Lower-left rectangle
        points = [[95, 170],
                  [95 - 75*sqrt(3)/2, 170 - 75/2],
                  [95 - 75*sqrt(3)/2 + 10/2, 170 - 75/2 - 10*sin(60*pi/180)],
                  [95 + 10/2, 170 - 10*sin(60*pi/180)]]
        points = [[points[i][0], points[i][1], points[(i + 1) % 4][0], points[(i + 1) % 4][1], i in [0, 3]]
                  for i in range(4)]
        shapes_to_draw.append(points)

        # Lower-right rhombus
        points = [[225, 190], [250, 175], [225, 160], [200, 175]]
        points = [[points[i][0], points[i][1], points[(i + 1) % 4][0], points[(i + 1) % 4][1], i in [0, 3]]
                  for i in range(4)]
        shapes_to_draw.append(points)

        # Use the half-planes defined above to draw polygons (not circle or ellipse)
        for i in range(len(shapes_to_draw)):
            shape_image = np.ones_like(self.baseImage, dtype=np.uint8)
            for j in range(len(shapes_to_draw[i])):
                x1 = shapes_to_draw[i][j][0]
                y1 = shapes_to_draw[i][j][1]
                x2 = shapes_to_draw[i][j][2]
                y2 = shapes_to_draw[i][j][3]
                above = shapes_to_draw[i][j][4]
                plane_image = np.array([[1 if row > col*(y2-y1)/(x2-x1) + (y1 - x1*(y2-y1)/(x2-x1)) else 0
                                         for col in range(self.width)] for row in range(self.height)], dtype=np.uint8)
                if above:
                    plane_image = 1 - plane_image
                shape_image = np.bitwise_and(shape_image, plane_image)
            self.baseImage = np.bitwise_or(self.baseImage, shape_image)

        # Define upper-right circle
        shape_image = np.array([[1 if (col - 225)**2 + (row - 50)**2 < 25**2 else 0
                               for col in range(self.width)] for row in range(self.height)], dtype=np.uint8)
        self.baseImage = np.bitwise_or(self.baseImage, shape_image)

        # Define middle ellipse
        shape_image = np.array([[1 if (col - 150) ** 2 + ((row - 100) * 2) ** 2 < 40 ** 2 else 0
                                 for col in range(self.width)] for row in range(self.height)], dtype=np.uint8)
        self.baseImage = np.bitwise_or(self.baseImage, shape_image)

        # Dilate the image based on the radius of the robot and the required clearance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.thickness*2 + 1, self.thickness*2 + 1))
        self.obstacleSpace = cv2.dilate(np.copy(self.baseImage), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1)
