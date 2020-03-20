from obstacleMap import ObstacleMap
from math import *
import numpy as np
import cv2
import sys
from datetime import datetime


class Robot:
    def __init__(self, start, goal, radius, clearance, rigid=False, play=False):
        """
        Initialization of the robot.
        :param start: starting coordinates for the robot, in tuple form (y, x)
        :param goal: goal coordinates for the robot, in tuple form (y, x)

        Attributes:
            map: Instance of the obstacle map to be navigated
            start: Same as init argument start
            goal: Same as init argument start
            openList: List of coordinates pending exploration, in the form: [(y, x), cost, action]
            openGrid: Matrix storing "1" for cells pending exploration, and "0" otherwise
            closeGrid: Matrix storing "1" for cells that have been explored, and "0" otherwise
            actionGrid: Matrix storing the optimal movement policy for cells that have been explored, and 255 otherwise
            backTrack: User-friendly visualization of the optimal path
        """
        self.actions = [[0, 1, 1],
                        [-1, 1, sqrt(2)],
                        [-1, 0, 1],
                        [-1, -1, sqrt(2)],
                        [0, -1, 1],
                        [1, -1, sqrt(2)],
                        [1, 0, 1],
                        [1, 1, sqrt(2)]]
        self.start = (199 - start[0], start[1])  # Starting coordinates in tuple form
        self.goal = (199 - goal[0], goal[1])  # Goal coordinates in tuple form
        self.success = False
        self.play = play

        # Handle radius and clearance arguments
        self.rigid = rigid
        self.radius = radius
        self.clearance = clearance
        if self.rigid:
            if self.radius < 0:
                sys.stdout.write("\nRadius is negative.  Exiting...\n")
                exit(0)
            elif self.clearance < 0:
                sys.stdout.write("\nClearance is negative.  Exiting...\n")
                exit(0)
            if self.radius == 0:
                sys.stdout.write("\nRadius is zero.  This is a point robot with clearance %d." % self.clearance)

        # Generate an instance of the map object
        self.map = ObstacleMap(radius + clearance)

        # Check to see if start and goal cells lie within map boundaries
        if not (0 <= self.start[0] < self.map.height) or not (0 <= self.start[1] < self.map.width):
            sys.stdout.write("\nStart lies outside of map boundaries!\n")
            exit(0)
        elif not (0 <= self.goal[0] < self.map.height) or not (0 <= self.goal[1] < self.map.width):
            sys.stdout.write("\nGoal lies outside of map boundaries!\n")
            exit(0)

        # Check to see if start and goal cells are in free spaces
        elif self.map.obstacleSpace[self.start]:
            sys.stdout.write("\nStart lies within obstacle space!\n")
            exit(0)
        elif self.map.obstacleSpace[self.goal]:
            sys.stdout.write("\nGoal lies within obstacle space!\n")
            exit(0)

        # Define cell maps to track exploration
        self.openList = []  # List of coordinates to be explored, in the form: [(y, x), cost, action]
        self.openGrid = np.zeros_like(self.map.baseImage, dtype=np.uint8)  # Grid of cells pending exploration
        self.closeGrid = np.zeros_like(self.map.baseImage, dtype=np.uint8)  # Grid of explored cells
        self.actionGrid = np.zeros_like(self.map.baseImage, dtype=np.uint8) + 255  # Grid containing movement policy
        self.backTrack = cv2.cvtColor(np.copy(self.map.baseImage)*255, cv2.COLOR_GRAY2RGB)  # Image of the optimal path
        self.pathImage = np.zeros_like(self.map.baseImage, dtype=np.uint8)  # Alternate image of optimal path
        self.frames = []
        self.solve()
        self.generate_video()

    def solve(self):
        """
        Solves the puzzle
        """
        # Initialize the open list/grid with the start cell
        self.openList = [[self.start, 0, 255]]  # [point, cost, action]
        self.openGrid[self.start] = 1
        sys.stdout.write("\nSearching for optimal path...\n")
        explored_count = 0
        free_count = int(np.sum(1 - self.map.obstacleSpace))
        start_time = datetime.today()
        while len(self.openList) > 0:
            # Find index of minimum cost cell
            cost_list = [self.openList[i][1] for i in range(len(self.openList))]
            index = int(np.argmin(cost_list, axis=0))
            cell = self.openList[index][0]
            cost = self.openList[index][1]
            action = self.openList[index][2]

            # See if goal cell has been reached
            if cell == self.goal:
                self.openList = []

            # Expand cell
            else:
                for a in range(len(self.actions)):
                    next_cell = (cell[0] + self.actions[a][0], cell[1]+self.actions[a][1])
                    # Check for map boundaries
                    if 0 <= next_cell[0] < self.map.height and 0 <= next_cell[1] < self.map.width:
                        # Check for obstacles
                        if not self.map.obstacleSpace[next_cell[0], next_cell[1]]:
                            # Check whether cell has been explored
                            if not self.closeGrid[next_cell[0], next_cell[1]]:
                                # Check if cell is already pending exploration
                                if not self.openGrid[next_cell[0], next_cell[1]]:
                                    self.openList.append([next_cell, cost + self.actions[a][2], a])
                                    self.openGrid[next_cell] = 1
                self.openList.pop(index)

            # Mark the cell as having been explored
            self.openGrid[cell] = 0
            self.closeGrid[cell] = 1
            self.actionGrid[cell] = action
            if explored_count % 15 == 0:
                self.frames.append(np.copy(self.closeGrid))
            explored_count += 1
            sys.stdout.write("\r%d out of %d cells explored (%.1f %%)" %
                             (explored_count, free_count, 100.0 * explored_count/free_count))
        current_cell = self.goal
        next_action_index = self.actionGrid[current_cell]

        # Output timing information
        end_time = datetime.today()
        duration = end_time - start_time
        sys.stdout.write("\n\nStart time:  %s\nEnd time:  %s" % (start_time, end_time))
        sys.stdout.write("\nRuntime:  ")
        sys.stdout.write(("%d hr, " % (duration.seconds // 3600)) if duration.seconds >= 3600 else "")
        sys.stdout.write(("%d min, " % ((duration.seconds // 60) % 3600))
                         if (duration.seconds // 60) % 3600 >= 1 else "")
        sys.stdout.write("%.3f sec" % ((duration.seconds % 60) + (duration.microseconds / 1000000.0)))

        # Check for failure to reach the goal cell
        if next_action_index == 255:
            if self.start != self.goal:
                sys.stdout.write("\n\nFailed to find a path to the goal!\n")
            else:
                sys.stdout.write("\n\nNo path generated.  Robot starts at goal space.\n")

        # Backtracking from the goal cell to extract an optimal path
        else:
            self.success = True
            sys.stdout.write("\n\nGoal reached!\n")
            while next_action_index != 255:
                current_cell = (current_cell[0] - self.actions[next_action_index][0],
                                current_cell[1] - self.actions[next_action_index][1])
                self.backTrack[current_cell] = (255, 0, 255)
                self.pathImage[current_cell] = 1
                next_action_index = self.actionGrid[current_cell]

    def generate_video(self):
        """
        Generates a video using H264 codec
        """
        sys.stdout.write("\nGenerating animation...\n")
        writer = cv2.VideoWriter('FinalAnimation.mp4', cv2.VideoWriter_fourcc('H', '2', '6', '4'), 30,
                                 (self.map.width, self.map.height))
        window_name = "Animation"

        # Add start frame to animation
        base_frame = cv2.cvtColor(np.zeros_like(self.map.baseImage, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        base_frame[:] = (192, 192, 192)
        base_frame = base_frame + cv2.cvtColor(self.map.baseImage, cv2.COLOR_GRAY2RGB) * 150
        first_frame = np.copy(base_frame)
        self.draw_start_and_goal(first_frame)
        for i in range(150):
            writer.write(first_frame)
        if self.play:
            cv2.imshow(window_name, first_frame)
            cv2.waitKey(5000)

        # Add exploration frames to animation
        back_color = np.zeros_like(self.backTrack, dtype=np.uint8)
        back_color[:] = (255, 192, 0)
        explore_frame = np.copy(base_frame)
        for frame in self.frames:
            explore_frame = np.copy(base_frame)
            explore_frame[np.where(frame)] = back_color[np.where(frame)]
            writer.write(explore_frame)
            if self.play:
                cv2.imshow(window_name, explore_frame)
                cv2.waitKey(15)

        # Add final frame to animation
        path_frame = np.copy(explore_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.radius * 2 + 1, self.radius * 2 + 1))
        path_mask = cv2.dilate(np.copy(self.pathImage), kernel)
        path_color = np.zeros_like(self.backTrack, dtype=np.uint8)
        path_color[:] = (255, 64, 64)
        path_frame[np.where(path_mask)] = path_color[np.where(path_mask)]
        self.draw_start_and_goal(path_frame)
        for i in range(60):
            writer.write(explore_frame)
        if self.play:
            cv2.imshow(window_name, explore_frame)
            cv2.waitKey(2000)
        for i in range(180):
            writer.write(path_frame if self.success else explore_frame)
        if self.play:
            cv2.imshow(window_name, path_frame if self.success else explore_frame)
            cv2.waitKey(5000)
        if self.play:
            cv2.destroyWindow(window_name)
        writer.release()

    def draw_start_and_goal(self, image):
        # Draw robot
        cv2.circle(image, (self.start[1], self.start[0]), self.radius + 3, (0, 0, 255), 1)
        cv2.circle(image, (self.start[1], self.start[0]), self.radius, (0, 0, 255), -1)

        # Draw goal
        cv2.line(image, (self.goal[1] - 5, self.goal[0] - 5), (self.goal[1] + 5, self.goal[0] + 5),
                 (0, 192, 0), 2)
        cv2.line(image, (self.goal[1] - 5, self.goal[0] + 5), (self.goal[1] + 5, self.goal[0] - 5),
                 (0, 192, 0), 2)
