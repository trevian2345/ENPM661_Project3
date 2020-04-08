from math import *
import numpy as np
import cv2
import sys
import argparse
from datetime import datetime
from obstacleMap import ObstacleMap


class Robot:
    """
    Robot class

    Attributes
    ----------
    start: Same as init argument start
    goal: Same as init argument goal
    openList: List of coordinates pending exploration, in the form: [(y, x, orientation), cost, action]
    openGrid: Matrix storing "1" for cells pending exploration, and "0" otherwise
    closeGrid: Matrix storing "1" for cells that have been explored, and "0" otherwise
    actionGrid: Matrix storing the optimal movement policy for cells that have been explored, and 255 otherwise
    """
    def __init__(self, start, goal, radius, clearance, step=1, theta_g=None, hw=None, play=False):
        """
        Initialization of the robot.
        :param start: starting coordinates for the robot, in tuple form (y, x, t)
        :param goal: goal coordinates for the robot, in tuple form (y, x)
        """
        self.res = 2.0  # Resolution of matrix for tracking duplicate states
        self.theta = 30  # Angle between action steps
        # Structure of self.actions:  (Distance, angle in units of self.theta, cost)
        self.actions = [[step, (i-2) % (360 // self.theta), 1] for i in range(5)]
        # Starting node in tuple form (y, x, orientation)
        self.start = (199 - start[1], start[0], (-int(start[2]) // self.theta) % (360 // self.theta))
        goal_2 = (-int(theta_g) // self.theta) % (360 // self.theta) if theta_g is not None else None
        self.goal = (199 - goal[1], goal[0], goal_2)  # Goal coordinates
        self.lastPosition = (-1, -1, -1)  # Coordinates of ending position (will be within certain tolerance of goal)
        self.success = True
        self.step = step
        self.goal_threshold = self.step * 1.5
        self.hw = hw if hw is not None else 2.0  # Heuristic weight (set to 1.0 for optimal path or 0.0 for Dijkstra)

        # Handle radius and clearance arguments
        self.radius = radius
        self.clearance = clearance
        t = self.radius + self.clearance
        self.map = ObstacleMap(t)
        if self.radius < 0:
            sys.stdout.write("\nRadius is negative.  Exiting...\n")
            exit(0)
        elif self.clearance < 0:
            sys.stdout.write("\nClearance is negative.  Exiting...\n")
            exit(0)
        if self.radius == 0:
            sys.stdout.write("\nRadius is zero.  This is a point robot with clearance %d." % self.clearance)

        # Check to see if start and goal cells lie within map boundaries
        if not (t <= self.start[0] < self.map.height - t) or not (t <= self.start[1] < self.map.width - t):
            sys.stdout.write("\nStart lies outside of map boundaries!\n")
            exit(0)
        elif not (t <= self.goal[0] < self.map.height - t) or not (t <= self.goal[1] < self.map.width - t):
            sys.stdout.write("\nGoal lies outside of map boundaries!\n")
            exit(0)

        # Check to see if start and goal cells are in free spaces
        elif self.map.is_colliding(self.start):
            sys.stdout.write("\nStart lies within obstacle space!\n")
            exit(0)
        elif self.map.is_colliding(self.goal):
            sys.stdout.write("\nGoal lies within obstacle space!\n")
            exit(0)

        # Define cell maps to track exploration
        self.openList = []  # List of coordinates to be explored, in the form: [(y, x, t), cost, action]
        self.configSpace = np.zeros((int(ceil(self.map.height * self.res)),
                                     int(ceil(self.map.width * self.res)), 360 // self.theta), dtype=np.uint8)
        self.openGrid = np.zeros_like(self.configSpace)  # Grid of cells pending exploration
        self.closeGrid = np.zeros_like(self.configSpace, dtype=np.uint8)  # Grid of explored cells
        # Grid containing parent cells
        self.parentGrid = np.zeros((self.configSpace.shape[0], self.configSpace.shape[1],
                                    self.configSpace.shape[2], 3), dtype=np.int) - 1
        # Grid containing movement policy
        self.actionGrid = np.zeros_like(self.parentGrid, np.float32)

        # Visualization image
        self.pathImage = np.zeros((self.configSpace.shape[0], self.configSpace.shape[1], 3),
                                  dtype=np.uint8)
        obstacle_space = np.array([[self.map.is_colliding((i / self.res, j / self.res), thickness=0)
                                    for j in range(self.configSpace.shape[1])]
                                   for i in range(self.configSpace.shape[0])], dtype=np.uint8)
        self.obstacleIndices = np.where(obstacle_space)
        self.freeIndices = np.where(1 - obstacle_space)
        self.freeCount = int(np.sum(1 - obstacle_space) * self.configSpace.shape[2])
        self.colors = {"free": (192, 192, 192),
                       "obstacle": (128, 128, 128),
                       "robot": (0, 0, 255),
                       "goal": (0, 192, 0),
                       "path": (64, 192, 255)}
        self.pathImage[self.obstacleIndices] = self.colors["obstacle"]
        self.pathImage[self.freeIndices] = self.colors["free"]
        self.baseImage = np.copy(self.pathImage)
        self.draw_robot_and_goal(self.start, self.pathImage)
        self.frames = [np.copy(self.pathImage)]
        self.play = play

        # Find a path
        self.solve()

    def solve(self):
        """
        Solves the puzzle
        """
        # Initialize the open list/grid with the start cell
        self.openList = [[self.start, 0]]  # [point, cost, action]
        self.openGrid[int(round(self.start[0] * self.res)), int(round(self.start[1] * self.res)), self.start[2]] = 1
        path_points = []
        sys.stdout.write("\nSearching for optimal path...\n")
        explored_count = 0
        start_time = datetime.today()
        while len(self.openList) > 0:
            # Find index of minimum cost cell
            cost_list = []
            for i in range(len(self.openList)):
                # Heuristic is the Euclidean distance to the goal
                ny = self.openList[i][0][0]
                nx = self.openList[i][0][1]
                heuristic = sqrt((ny - self.goal[0]) ** 2 + (nx - self.goal[1]) ** 2) * self.hw
                cost_list.append(self.openList[i][1] + heuristic)
            index = int(np.argmin(cost_list, axis=0))
            cell = self.openList[index][0]
            cost = self.openList[index][1]

            # See if goal cell has been reached (with threshold condition)
            if self.on_goal(cell):
                self.lastPosition = cell
                self.openList = []

            # Expand cell
            else:
                for a in range(len(self.actions)):
                    next_cell = (cell[0] + self.actions[a][0] * sin(self.theta*(self.actions[a][1] + cell[2])*pi/180),
                                 cell[1] + self.actions[a][0] * cos(self.theta*(self.actions[a][1] + cell[2])*pi/180),
                                 (cell[2] + self.actions[a][1]) % (360 // self.theta))
                    ny, nx, nt = next_cell

                    # Check for map boundaries
                    if 0 <= ny < self.map.height and 0 <= nx < self.map.width:
                        # Check for obstacles
                        if not self.map.is_colliding((ny, nx), check_inside=(self.radius+self.clearance <= self.step)):
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

                self.openList.pop(index)
                if len(self.openList) == 0:
                    self.success = False

            # Mark the cell as having been explored
            self.openGrid[int(cell[0] * self.res), int(cell[1] * self.res), cell[2]] = 0
            self.closeGrid[int(cell[0] * self.res), int(cell[1] * self.res), cell[2]] = 1

            # Update visualization
            line_color = np.sum(self.closeGrid[int(cell[0] * self.res), int(cell[1] * self.res)])
            line_color = (255, int(255 - (1.0 - line_color / (360 / self.theta)) * 192), 64)
            if explored_count > 0:
                parent_cell = tuple(self.parentGrid[int(cell[0] * self.res), int(cell[1] * self.res), cell[2]][:2])
                parent_cell = (parent_cell[1], parent_cell[0])
                cv2.line(self.pathImage, (int(cell[1] * self.res), int(cell[0] * self.res)),
                         parent_cell, line_color)
            if explored_count % 10 == 0 or len(self.openList) == 0 and self.success:  # Display every 10 frames
                self.frames.append(np.copy(self.pathImage))

            explored_count += 1
            if explored_count % 100 == 0:
                sys.stdout.write("\r%d out of %d cells explored (%.1f %%)" %
                                 (explored_count, self.freeCount, 100.0 * explored_count / self.freeCount))

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
        if self.on_goal(self.start):
            sys.stdout.write("\n\nNo path generated.  Robot starts at goal space.\n")
        elif not self.success:
            sys.stdout.write("\n\nFailed to find a path to the goal!\n")

        # Backtracking from the goal cell to extract an optimal path
        else:
            sys.stdout.write("\n\nGoal reached!  If you receive an error below related to FFmpeg, but the output "
                             "video plays, you may disregard the error.\n\n")
            goal_y = int(self.lastPosition[0] * self.res)
            goal_x = int(self.lastPosition[1] * self.res)
            goal_r = self.lastPosition[2]
            current_cell = (goal_y, goal_x, goal_r)
            next_cell = tuple(self.parentGrid[current_cell])
            path_points = [(int(self.lastPosition[0]), int(self.lastPosition[1]), self.lastPosition[2])]
            while sum(next_cell) >= 0:
                cv2.line(self.pathImage, (current_cell[1], current_cell[0]),
                         (next_cell[1], next_cell[0]), self.colors["path"],
                         thickness=int(1 + 2 * self.radius * self.res))
                path_points.append(self.actionGrid[current_cell])
                current_cell = next_cell
                next_cell = tuple(self.parentGrid[next_cell])
            path_points.reverse()
            self.frames.append(np.copy(self.pathImage))
            self.draw_robot_and_goal(self.start, self.pathImage, start_only=True)

        # Create output video
        writer = cv2.VideoWriter('FinalAnimation.mp4', cv2.VideoWriter_fourcc('H', '2', '6', '4'), 30,
                                 (int(self.map.width * self.res), int(self.map.height * self.res)))
        window_name = "Animation"
        for i in range(150):
            writer.write(self.frames[0])
        for frame in self.frames[:len(self.frames)-1]:
            writer.write(frame)
            if self.play:
                cv2.imshow(window_name, frame)
                cv2.waitKey(15)
        for i in range(60):
            writer.write(self.frames[len(self.frames) - 2])
        if self.play:
            cv2.imshow(window_name, self.frames[len(self.frames) - 2])
            cv2.waitKey(2000)
        for i in range(180):
            writer.write(self.pathImage if self.success else self.frames[len(self.frames) - 2])
        if self.play:
            cv2.imshow(window_name, self.pathImage if self.success else self.frames[len(self.frames) - 2])
            cv2.waitKey(5000)
        # Follow the path
        if self.success:
            next_image = np.copy(self.baseImage)
            for i in range(len(path_points)):
                next_image = np.copy(self.baseImage)
                self.draw_robot_and_goal(path_points[i], next_image, rim=False)
                for j in range(min(1 + int(self.step) // 5, 5)):
                    writer.write(next_image)
            for i in range(150):
                writer.write(next_image)
            if self.play:
                cv2.imshow(window_name, next_image)
                cv2.waitKey(15)
        writer.release()
        cv2.destroyWindow(window_name)

    def on_goal(self, point):
        result = sqrt((self.goal[0] - point[0]) ** 2 + (self.goal[1] - point[1]) ** 2) <= self.goal_threshold
        return result and (True if self.goal[2] is None else self.goal[2] == point[2])

    def draw_robot_and_goal(self, robot_pos, image, start_only=False, rim=True):
        """
        Draws the robot and goal positions.

        Parameters
        ----------
        robot_pos: tuple
            Position at which to draw the robot
        image: np.array
            Image on which to draw the robot and goal
        start_only: bool, optional
            Whether to draw the start position or not.  Defaults to False.
        rim: bool, optional
            Whether to draw the rim around the robot's start position.  Defaults to True.  The goal always has a rim.
        """
        points = [(self.goal, self.colors["goal"])] if not start_only else []
        points.append((robot_pos, self.colors["robot"]))
        goal = len(points) - 1
        for pt, col in points:
            rad = int(self.radius + (self.goal_threshold if goal else 0.0))
            if rim or goal:
                cv2.circle(image, (int(pt[1] * self.res), int(pt[0] * self.res)), int(rad * self.res + 4),
                           col, 1)
            cv2.circle(image, (int(pt[1] * self.res), int(pt[0] * self.res)), int(rad * self.res),
                       col, -1)
            if pt[2] is not None:
                s_arrow = [[int((pt[1]+(rad+2)*cos((pt[2]*self.theta + 45)*pi/180)) * self.res),
                            int((pt[0]+(rad+2)*sin((pt[2]*self.theta + 45)*pi/180)) * self.res)],
                           [int((pt[1]+(rad+2)*1.4*cos(pt[2]*self.theta*pi/180)) * self.res),
                            int((pt[0]+(rad+2)*1.4*sin(pt[2]*self.theta*pi/180)) * self.res)],
                           [int((pt[1]+(rad+2)*cos((pt[2]*self.theta - 45)*pi/180)) * self.res),
                            int((pt[0]+(rad+2)*sin((pt[2]*self.theta - 45)*pi/180)) * self.res)]]
                polygon = np.array(s_arrow, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [polygon], False, col, thickness=1)
            goal -= 1
        image[self.obstacleIndices] = self.colors["obstacle"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Project-3 -- Phase-2:  Navigation of a rigid robot from a start point"
                    "to an end point using A* algorithm.")
    parser.add_argument('initialX', type=float, help='X-coordinate of initial node of the robot')
    parser.add_argument('initialY', type=float, help='Y-coordinate of initial node of the robot')
    parser.add_argument('theta_s', type=float, help='Start point orientation of the robot')
    parser.add_argument('goalX', type=float, help='X-coordinate of goal node of the robot')
    parser.add_argument('goalY', type=float, help='Y-coordinate of goal node of the robot')
    parser.add_argument('radius', type=float, help='Indicates the radius of the rigid robot')
    parser.add_argument("clearance", type=float,
                        help="Indicates the minimum required clearance between the rigid robot and obstacles")
    parser.add_argument('step', type=float, help='Indicates the step size of the robot')
    parser.add_argument('--theta_g', type=float, help='Goal point orientation of the robot.  Omit for any orientation.')
    parser.add_argument('--hw', type=float, help='Heuristic weight.  Defaults to 2.0 when omitted. '
                                                 '(Set to 1.0 for optimal path or 0.0 for Dijkstra)')
    parser.add_argument('--play', action="store_true", help="Play using opencv's imshow")
    args = parser.parse_args()

    sx = args.initialX
    sy = args.initialY
    ts = args.theta_s
    gx = args.goalX
    gy = args.goalY
    tg = args.theta_g
    h_weight = args.hw
    r = args.radius
    c = args.clearance
    s = args.step
    p = args.play
    start_pos = (sx, sy, ts)
    goal_pos = (gx, gy)
    rigidRobot = Robot(start_pos, goal_pos, r, c, s, theta_g=tg, hw=h_weight, play=p)
