from math import *
import numpy as np
import sys
import argparse
from datetime import datetime
from obstacleMap import ObstacleMap


class Robot:
    def __init__(self, start, goal, radius, clearance, step=1):
        """
        Initialization of the robot.
        :param start: starting coordinates for the robot, in tuple form (y, x, t)
        :param goal: goal coordinates for the robot, in tuple form (y, x)
        Attributes:
            start: Same as init argument start
            goal: Same as init argument start
            openList: List of coordinates pending exploration, in the form: [(y, x, orientation), cost, action]
            openGrid: Matrix storing "1" for cells pending exploration, and "0" otherwise
            closeGrid: Matrix storing "1" for cells that have been explored, and "0" otherwise
            actionGrid: Matrix storing the optimal movement policy for cells that have been explored, and 255 otherwise
            backTrack: User-friendly visualization of the optimal path
        """
        self.res = 2  # Resolution of matrix for tracking duplicate states
        self.theta = 30  # Angle between action steps
        # Structure of self.actions:  (Distance, angle in units of self.theta, cost)
        self.actions = [[step, (i-2) % (360 // self.theta), 1] for i in range(5)]
        # Starting node in tuple form (y, x, orientation)
        self.start = (199 - start[1], start[0], (start[2] // self.theta) % (360 // self.theta))
        self.goal = (199 - goal[1], goal[0], None)  # Goal coordinates in tuple form
        self.goal_threshold = 1.5
        self.success = False
        self.step = step

        # Handle radius and clearance arguments
        self.radius = radius
        self.clearance = clearance
        self.map = ObstacleMap(self.radius + self.clearance)
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
        self.configSpace = np.zeros((int(ceil(self.map.height * self.res)),
                                     int(ceil(self.map.width * self.res)), 360 // self.theta), dtype=np.uint8)
        self.openGrid = np.zeros_like(self.configSpace)  # Grid of cells pending exploration
        self.closeGrid = np.zeros_like(self.configSpace, dtype=np.uint8)  # Grid of explored cells
        # Grid containing parent cells
        self.parentGrid = np.zeros((self.configSpace.shape[0], self.configSpace.shape[1],
                                    self.configSpace.shape[2], 3), dtype=np.int)
        # Grid containing movement policy
        self.actionGrid = np.zeros_like(self.parentGrid, np.float32)
        print("New Shape: ", self.parentGrid.shape)
        self.pathImage = np.zeros_like(self.configSpace, dtype=np.uint8)  # Alternate image of optimal path
        self.frames = []
        self.solve()

    def solve(self):
        """
        Solves the puzzle
        """
        # Initialize the open list/grid with the start cell
        self.openList = [[self.start, 0]]  # [point, cost, action]
        self.openGrid[self.start[0] * self.res, self.start[1] * self.res, self.start[2]] = 1
        sys.stdout.write("\nSearching for optimal path...\n")
        explored_count = 0
        free_count = int(np.sum(1 - self.configSpace))
        start_time = datetime.today()
        while len(self.openList) > 0:
            # Find index of minimum cost cell
            cost_list = []
            for i in range(len(self.openList)):
                # Heuristic is the Euclidean distance to the goal
                ny = self.openList[i][0][0]
                nx = self.openList[i][0][1]
                heuristic = sqrt((ny - self.goal[0]) ** 2 + (nx - self.goal[1]) ** 2)
                cost_list.append(self.openList[i][1] + heuristic)
            index = int(np.argmin(cost_list, axis=0))
            cell = self.openList[index][0]
            cost = self.openList[index][1]

            # See if goal cell has been reached (with threshold condition)
            if self.on_goal(cell):
                self.goal = cell
                self.openList = []

            # Expand cell
            else:
                for a in range(len(self.actions)):
                    next_cell = (cell[0] + self.actions[a][0] * sin(self.theta*(self.actions[a][1] + cell[2])*pi/180),
                                 cell[1] + self.actions[a][0] * cos(self.theta*(self.actions[a][1] + cell[2])*pi/180),
                                 (cell[2] + self.actions[a][2]) % (360 // self.theta))
                    ny, nx, nt = next_cell
                    # print(ny)
                    # print(nx)
                    # print(nt)
                    # Check for map boundaries
                    if 0 <= ny < self.map.height and 0 <= nx < self.map.width:
                        # Check for obstacles
                        if not self.map.is_colliding((ny, nx)):
                            # Check whether cell has been explored
                            if not self.closeGrid[int(ny * self.res), int(nx * self.res), nt]:
                                # Check if cell is already pending exploration
                                if not self.openGrid[int(ny * self.res), int(nx * self.res), nt]:
                                    self.openList.append([next_cell, cost + self.step])
                                    parent = [int(cell[0] * self.res), int(cell[1] * self.res), cell[2]]
                                    self.parentGrid[int(ny * self.res), int(nx * self.res), nt] = parent
                                    action = [cell[0], cell[1], cell[2]]
                                    self.actionGrid[int(ny * self.res), int(nx * self.res), nt] = action
                                    self.openGrid[int(ny * self.res), int(nx * self.res), nt] = 1
                                else:
                                    # TODO:  Handle cell being approached from two cells with same cumulative cost
                                    pass
                self.openList.pop(index)

            # Mark the cell as having been explored
            self.openGrid[int(cell[0] * self.res), int(cell[1] * self.res), cell[2]] = 0
            self.closeGrid[int(cell[0] * self.res), int(cell[1] * self.res), cell[2]] = 1
            # if explored_count % 15 == 0:
            #     self.frames.append(np.copy(self.closeGrid))
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
            if not self.on_goal(self.start):
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

    @staticmethod
    def handling_theta(t):
        """
        Function to handle the theta values(to nearest 30)
        :param t: input theta
        :return: theta normalized to an integer from 0 to 11
        """
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

    @staticmethod
    def handling_point(point):
        """
        Function to round coordinates
        :param point: Coordinate to round
        :return: Rounded point
        """
        new = ceil(point)
        # print(0)
        # print(new)
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

    def on_goal(self, point):
        return sqrt((self.goal[0] - point[0]) ** 2 + (self.goal[1] - point[1]) ** 2) <= self.goal_threshold


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
