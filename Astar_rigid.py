from math import *
import numpy as np
import cv2
import sys
import argparse
from datetime import datetime

class Robot:
    def __init__(self, start, goal, radius, clearance, step=1):
        """
                Initialization of the robot.
                :param start: starting coordinates for the robot, in tuple form (y, x, t)
                :param goal: goal coordinates for the robot, in tuple form (y, x)
                Attributes:
                    start: Same as init argument start
                    goal: Same as init argument start
                    openList: List of coordinates pending exploration, in the form: [(y, x), cost, action]
                    openGrid: Matrix storing "1" for cells pending exploration, and "0" otherwise
                    closeGrid: Matrix storing "1" for cells that have been explored, and "0" otherwise
                    actionGrid: Matrix storing the optimal movement policy for cells that have been explored, and 255 otherwise
                    backTrack: User-friendly visualization of the optimal path
                """
        self.actions = [[step*1, 0, 0],
                        [step*cos(30),  step*sin(30), 1],
                        [step*cos(60),  step*sin(60), 2],
                        [step*cos(330), step*sin(330), 11],
                        [step*cos(300), step*sin(300), 10]]
        self.start = (199 - start[0], start[1], self.handling_theta(start[2]))  # Starting node in tuple form
        self.goal = (199 - goal[0], goal[1])  # Goal coordinates in tuple form
        self.success = False
        self.step = step

        # Handle radius and clearance arguments

        self.radius = radius
        self.clearance = clearance
        if True:
            if self.radius < 0:
                sys.stdout.write("\nRadius is negative.  Exiting...\n")
                exit(0)
            elif self.clearance < 0:
                sys.stdout.write("\nClearance is negative.  Exiting...\n")
                exit(0)
            if self.radius == 0:
                sys.stdout.write("\nRadius is zero.  This is a point robot with clearance %d." % self.clearance)

        # Define cell maps to track exploration
        self.openList = []  # List of coordinates to be explored, in the form: [(y, x, t), cost, action]
        self.configSpace = np.zeros((400, 600, 12), dtype=np.uint8)
        self.openGrid = np.zeros_like(self.configSpace, dtype=np.uint8)  # Grid of cells pending exploration
        self.closeGrid = np.zeros_like(self.configSpace, dtype=np.uint8)  # Grid of explored cells
        self.actionGrid = np.zeros_like(self.configSpace, dtype=np.uint8) + 255  # Grid containing movement policy
        #self.backTrack = cv2.cvtColor(np.copy(self.configSpace) * 255,
                                     # cv2.COLOR_GRAY2RGB)  # Image of the optimal path
        self.pathImage = np.zeros_like(self.configSpace, dtype=np.uint8)  # Alternate image of optimal path
        self.frames = []
        self.solve()

    def solve(self):
            """
            Solves the puzzle
            """
            # Initialize the open list/grid with the start cell
            self.openList = [[self.start, 0, 255]]  # [point, cost, action]
            self.openGrid[2*self.start[0], 2*self.start[1], self.start[2]] = 1
            sys.stdout.write("\nSearching for optimal path...\n")
            explored_count = 0
            free_count = int(np.sum(1 - self.configSpace))
            start_time = datetime.today()
            while len(self.openList) > 0:
                # Find index of minimum cost cell
                cost_list = [self.openList[i][1] for i in range(len(self.openList))]
                index = int(np.argmin(cost_list, axis=0))
                cell = self.openList[index][0]
                cost = self.openList[index][1]
                action = self.openList[index][2]

                # See if goal cell has been reached(with threshold condition)
                if sqrt((self.goal[0] - cell[1])**2 + (self.goal[1] - cell[0])**2) <= 1.5:
                    self.openList = []

                # Expand cell
                else:
                    for a in range(len(self.actions)):
                        new_cell = (
                        cell[0] + self.actions[a][0], cell[1] + self.actions[a][1], cell[2] + self.actions[a][2])
                        if new_cell[2] > 11:
                            theta = new_cell[2] - 11
                        else:
                            theta = new_cell[2]
                        next_cell = (self.handling_point(new_cell[0]), self.handling_point(new_cell[1]), theta)
                        # print(next_cell[0])
                        # print(next_cell[1])
                        # print(next_cell[2])
                        # Check for map boundaries
                        if 0 <= next_cell[0] < 200 and 0 <= next_cell[1] < 300:
                            # Check for obstacles
                            # if not self.obstacleSpace[next_cell[0], next_cell[1]]:
                            # Check whether cell has been explored
                            if not self.closeGrid[2*next_cell[0], 2*next_cell[1], next_cell[2]]: #(Error)
                                # Check if cell is already pending exploration
                                if not self.openGrid[2*next_cell[0], 2*next_cell[1], next_cell[2]]:
                                    heuristic = abs(next_cell[0] - self.goal[0]) + abs(next_cell[1] - self.goal[1])
                                    self.openList.append([next_cell, cost + self.step + heuristic, self.actions[a][2]])
                                    self.openGrid[2*next_cell[0], 2*next_cell[1], next_cell[2]] = 1
                    self.openList.pop(index)

                # Mark the cell as having been explored
                self.openGrid[2*cell[0], 2*cell[1], cell[2]] = 0
                self.closeGrid[2*cell[0], 2*cell[1], cell[2]] = 1
                self.actionGrid[2*cell[0], 2*cell[1], cell[2]] = action
                if explored_count % 15 == 0:
                    self.frames.append(np.copy(self.closeGrid))
                explored_count += 1
                sys.stdout.write("\r%d out of %d cells explored (%.1f %%)" %
                                 (explored_count, free_count, 100.0 * explored_count / free_count))
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
                pass
                # self.success = True
                # sys.stdout.write("\n\nGoal reached!\n")
                # while next_action_index != 255:
                #     current_cell = (current_cell[0] - self.actions[next_action_index][0],
                #                     current_cell[1] - self.actions[next_action_index][1])
                #     self.backTrack[current_cell] = (255, 0, 255)
                #     self.pathImage[current_cell] = 1
                #     next_action_index = self.actionGrid[current_cell]

        # Function to handle the theta values(to nearest 30)
    def handling_theta(self, t):
            if (30 > t >= 15) or (330 <= t < 3455):
                return 1
            elif 0 <= t < 3455 or 0 > t >= 345:
                return 0
            else:
                t = t / 30
                t_n = t % floor(t)
                if t_n <= 0.5:
                    t = floor(t)
                else:
                    t = ceil(t)
                return t

        # Function to handle co-ordinates(rounding)
    def handling_point(self, point):
            new = ceil(point)
            print(0)
            print(new)
            if 0.75 >= new - point >= 0.25:
                point = new - 0.5
                print(1)
            elif 0.75 < new - point <= 1:
                point = floor(point)
                print(2)
            elif 0.25 > new - point >= 0:
                point = new
                print(3)
            else:
                return point
            return point

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Project-3 -- Phase-2:  Navigation of a rigid robot from a start point"
                    "to an end point using A* algorithm.")
    parser.add_argument('initialX', type=int, help='X-coordinate of initial node of the robot')
    parser.add_argument('initialY', type=int, help='Y-coordinate of initial node of the robot')
    parser.add_argument('theta_s', type=int, help='Start point orientation of the robot')
    parser.add_argument('goalX', type=int, help='X-coordinate of goal node of the robot')
    parser.add_argument('goalY', type=int, help='Y-coordinate of goal node of the robot')
    # parser.add_argument('theta_g', type=int, help='Goal point orientation of the robot')
    parser.add_argument('radius', type=int, help='Indicates the radius of the rigid robot')
    parser.add_argument("clearance", type=int,
                        help="Indicates the minimum required clearance between the rigid robot and obstacles")
    parser.add_argument('step', type=int, help='Indicates the step size of the robot')
    # parser.add_argument('--play', action="store_true", help="Play using opencv's imshow")
    args = parser.parse_args()

    sx = args.initialX
    sy = args.initialY
    ts = args.theta_s
    gx = args.goalX
    gy = args.goalY
    # tg = args.theta_g
    r = args.radius
    c = args.clearance
    s = args.step
    # p = args.play
    start_pos = (sx, sy, ts)
    goal_pos = (gx, gy)
    rigidRobot = Robot(start_pos, goal_pos, r, c, s)

