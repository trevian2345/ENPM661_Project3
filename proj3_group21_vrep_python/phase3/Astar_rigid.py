from math import *
import numpy as np
import cv2
import sys
import argparse
from datetime import datetime
from obstacleMap import ObstacleMap


class Robot:
    """
    Robot class.

    Attributes
    ----------
    start: tuple of float
        Same as init argument start
    goal: tuple of float
        Same as init argument goal
    openList: np.array
        List of coordinates pending exploration, in the form: [(y, x, orientation), cost, action]
    openGrid: np.array
        Matrix storing "1" for cells pending exploration, and "0" otherwise
    closeGrid: np.array
        Matrix storing "1" for cells that have been explored, and "0" otherwise
    actionGrid: np.array
        Matrix storing the optimal movement policy for cells that have been explored, and 255 otherwise
    i_res: float
        Resolution of images for generating visualization
    """
    def __init__(self, start, goal, rpm1, rpm2, clearance=None, hw=None, reduced=False, play=False):
        """
        Initialization of the robot.

        Parameters
        ----------
        start: tuple of float
            starting coordinates for the robot, in tuple form (y, x, t)
        goal: tuple of float
            goal coordinates for the robot, in tuple form (y, x)
        hw: float, optional
            Heuristic weight
        play: bool, optional
            Whether to play using OpenCV's imshow()
        """
        self.res = 0.125  # Resolution of matrix for tracking duplicate states
        self.i_res = 0.025  # Resolution for visualizations
        self.theta = 30  # Resolution of robot orientation for differentiation of action steps
        self.dt = 1.0  # Time between actions
        self.radius = 0.177  # Robot radius (see data sheet)
        self.w_axis = self.radius * 0.8  # Length of axis between wheels (assumed)
        self.wr = 0.038  # Wheel radius, in meters (see data sheet)
        self.rpm1 = rpm1
        self.rpm2 = rpm2
        self.clearance = clearance if clearance is not None else 0.08  # 8 cm clearance, just in case
        if self.clearance < 0:
            sys.stdout.write("\nClearance is negative.  Exiting...\n")
            exit(0)
        self.map = ObstacleMap(self.radius + self.clearance)

        # Handle RPM arguments
        if self.rpm1 < 0:
            sys.stdout.write("\nRPM1 is negative.  Exiting...\n")
            exit(0)
        elif self.rpm2 < 0:
            sys.stdout.write("\nRPM2 is negative.  Exiting...\n")
            exit(0)

        # Structure of self.actions:  [rpm of left wheel, rpm of right wheel]
        self.actions = [[0, rpm1],
                        [rpm1, 0],
                        [rpm1, rpm1],
                        [0, rpm2],
                        [rpm2, 0],
                        [rpm2, rpm2],
                        [rpm1, rpm2],
                        [rpm2, rpm1]][(5 if reduced else 0):]

        # Starting node in tuple form (y, x, orientation)
        self.start = (self.map.height - start[1], start[0], (-start[2] * pi / 180.0) % (2.0 * pi))
        self.goal = (self.map.height - goal[1], goal[0])  # Goal coordinates
        self.lastPosition = (-1, -1, -1)  # Coordinates of ending position (will be within certain tolerance of goal)
        self.success = True

        # Goal threshold:  minimum distance that can be covered in one action step
        self.goal_threshold = min(rpm1, rpm2) * 2.0 * pi / 60.0 * self.dt * self.wr * 0.2

        # Heuristic weight (set to <= 1.0 for optimal path or 0.0 for Dijkstra)
        self.hw = hw if hw is not None else 1.0

        # Output arguments
        sys.stdout.write("\nThe following parameters have been provided:")
        sys.stdout.write("\n    Robot start:  x = %.2f, y = %.2f, theta = %d degrees" %
                         (start[0], start[1], (int(-self.start[2] * 180.0 / pi) % 360)))
        sys.stdout.write("\n    Robot goal:  x = %.2f, y = %.2f" % (goal[0], goal[1]))
        sys.stdout.write("\n    Wheel RPMs:  %.2f and %.2f" % (self.rpm1, self.rpm2))
        sys.stdout.write("\n    Clearance:  %.2f meters" % self.clearance)
        sys.stdout.write("\n    Heuristic weight:  %.2f\n" % self.hw)

        # Check to see if start and goal cells lie within map boundaries
        t = self.radius + self.clearance
        if not (t <= self.start[0] < self.map.height - t) or not (t <= self.start[1] < self.map.width - t):
            sys.stdout.write("\nStart lies outside of map boundaries!\n")
            exit(0)
        elif not (t <= self.goal[0] < self.map.height - t) or not (t <= self.goal[1] < self.map.width - t):
            sys.stdout.write("\nGoal lies outside of map boundaries!\n")
            exit(0)

        # Check to see if start and goal cells are in free spaces
        if self.map.collision_circle((self.start[0], self.start[1], self.radius)):
            sys.stdout.write("\nStart lies within obstacle space!\n")
            exit(0)
        if self.map.collision_circle((self.goal[0], self.goal[1], self.radius)):
            sys.stdout.write("\nGoal lies within obstacle space!\n")
            exit(0)

        # Define cell maps to track exploration
        self.openList = []  # List of coordinates to be explored, in the form: [(y, x, t), cost, action]
        self.configSpace = np.zeros((int(ceil(self.map.height / self.res)),
                                     int(ceil(self.map.width / self.res)), 360 // self.theta), dtype=np.uint8)

        # Grid of cells pending exploration
        self.openGrid = np.zeros_like(self.configSpace)

        # Grid of explored cells
        self.closeGrid = np.zeros_like(self.configSpace, dtype=np.uint8)

        # Grid containing parent cells
        self.parentGrid = np.zeros((self.configSpace.shape[0], self.configSpace.shape[1],
                                    self.configSpace.shape[2], 3), dtype=np.int) - 1
        # Grid containing movement policy
        self.actionGrid = np.zeros((self.configSpace.shape[0], self.configSpace.shape[1],
                                    self.configSpace.shape[2], 9), dtype=np.float32)

        # State matrices
        sys.stdout.write("\nCreating state space matrices and base illustrations...")
        blocked_states = np.array([[self.map.collision_point((i * self.res, j * self.res))
                                    for j in range(self.configSpace.shape[1])]
                                   for i in range(self.configSpace.shape[0])], dtype=np.uint8)
        self.obstacleIndices = np.where(blocked_states)
        self.freeCount = int(np.sum(1 - blocked_states) * self.configSpace.shape[2])

        # Visualization images
        self.imageShape = (int(round(self.map.height / self.i_res)), int(round(self.map.width / self.i_res)), 3)
        self.pathImage = np.zeros(self.imageShape, dtype=np.uint8)
        blocked_pixels = np.array([[self.map.collision_point((i * self.i_res, j * self.i_res))
                                    for j in range(self.imageShape[1])]
                                   for i in range(self.imageShape[0])], dtype=np.uint8)
        self.obstaclePixels = np.where(blocked_pixels)
        self.colors = {"free": (192, 192, 192),
                       "obstacle": (128, 128, 128),
                       "robot": (0, 0, 255),
                       "goal": (0, 192, 0),
                       "path": (64, 192, 255)}
        self.pathImage[np.where(1 - blocked_pixels)] = self.colors["free"]
        self.baseImage = np.copy(self.pathImage)
        self.draw_goal(self.goal, self.pathImage)
        self.pathImage[np.where(blocked_pixels)] = self.colors["obstacle"]
        self.draw_robot(self.start, self.pathImage)
        self.frames = [np.copy(self.pathImage)]
        self.play = play

        # Array for velocities
        self.velocityArray = []

        # Find a path
        self.solve()

    def solve(self):
        """
        Solves the puzzle.
        """
        # Initialize the open list/grid with the start cell
        self.openList = [[self.start, 0, -1]]  # [point, cost, arc]
        self.openGrid[int(self.start[0] / self.res), int(self.start[1] / self.res),
                      int(self.start[2] * 180.0 / pi) // self.theta] = 1
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
            arc_to_draw = self.openList[index][2]
            cell_theta_norm = int(cell[2] * 180.0 / pi) // self.theta

            # See if goal cell has been reached (with threshold condition)
            if self.on_goal(cell):
                self.lastPosition = cell
                self.openList = []

            # Expand cell
            else:
                for a in range(len(self.actions)):
                    next_cell = self.next_position(cell, self.actions[a][0], self.actions[a][1])
                    ny, nx, nt, arc_length, arc_center, turn = next_cell
                    theta_norm = int(nt * 180.0 / pi) // self.theta

                    # Check for map boundaries
                    if 0.0 <= ny < self.map.height and 0.0 <= nx < self.map.width:
                        # Check for obstacles
                        collision = self.map.collision_circle((ny, nx, self.radius + self.clearance))
                        collision = collision or self.map.collision_circle(((ny + cell[0]) / 2.0, (nx + cell[1]) / 2.0,
                                                                            self.radius + self.clearance * 1.5))
                        if not collision:
                            # Check whether cell has been explored
                            if not self.closeGrid[int(ny / self.res), int(nx / self.res), theta_norm]:
                                # Check if cell is already pending exploration
                                if not self.openGrid[int(ny / self.res), int(nx / self.res), theta_norm]:
                                    self.openList.append([(ny, nx, nt), cost + arc_length, arc_center])
                                    parent = [int(cell[0] / self.res), int(cell[1] / self.res), cell_theta_norm]
                                    self.parentGrid[int(ny / self.res), int(nx / self.res), theta_norm] = parent
                                    action = [cell[0], cell[1], cell[2]]
                                    if arc_center == -1:
                                        action.extend([-1] * (self.actionGrid.shape[3] - len(action)))
                                    else:
                                        action.extend(list(arc_center))
                                        action.append(turn)
                                    self.actionGrid[int(ny / self.res), int(nx / self.res), theta_norm] = action
                                    self.openGrid[int(ny / self.res), int(nx / self.res), theta_norm] = 1

                self.openList.pop(index)
                if len(self.openList) == 0:
                    self.success = False

            # Mark the cell as having been explored
            self.openGrid[int(cell[0] / self.res), int(cell[1] / self.res), cell_theta_norm] = 0
            self.closeGrid[int(cell[0] / self.res), int(cell[1] / self.res), cell_theta_norm] = 1

            # Update visualization
            if explored_count > 0:
                line_color = np.sum(self.closeGrid[int(cell[0] / self.res), int(cell[1] / self.res)])
                line_color = (255, int(255 - (1.0 - line_color / (360 / self.theta)) * 192), 64)
                parent_cell = tuple(self.parentGrid[int(cell[0] / self.res), int(cell[1] / self.res),
                                                    cell_theta_norm][:2])
                parent_cell = (int(parent_cell[1] * self.res / self.i_res), int(parent_cell[0] * self.res / self.i_res))

                # Draw a line or an arc from the parent point to the current point
                if arc_to_draw == -1:
                    cv2.line(self.pathImage, (int(cell[1] / self.i_res), int(cell[0] / self.i_res)),
                             parent_cell, line_color)
                else:
                    arc_y, arc_x, arc_r, arc_th1, arc_th2 = arc_to_draw
                    if arc_th2 < arc_th1:
                        arc_th2 += 360
                    cv2.ellipse(self.pathImage, (int(arc_x), int(arc_y)), (int(arc_r), int(arc_r)), 0,
                                arc_th1, arc_th2, line_color)
            if explored_count % 5 == 0 or (len(self.openList) == 0 and self.success):  # Display every 5 frames
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
            sys.stdout.write("\n\nGoal reached!\n")
            goal_y = int(self.lastPosition[0] / self.res)
            goal_x = int(self.lastPosition[1] / self.res)
            goal_r = int(self.lastPosition[2] * 180.0 / pi) // self.theta
            current_cell = (goal_y, goal_x, goal_r)
            current_theta = self.lastPosition[2]
            current_actual = self.lastPosition
            next_cell = tuple(self.parentGrid[current_cell])
            path_points = [[self.lastPosition[0], self.lastPosition[1], self.lastPosition[2]]]
            self.draw_goal(self.goal, self.pathImage)
            cell_split = 10
            while sum(next_cell) >= 0:
                if current_cell[2] == next_cell[2]:
                    cv2.line(self.pathImage, (int(current_cell[1] * self.res / self.i_res),
                                              int(current_cell[0] * self.res / self.i_res)),
                             (int(next_cell[1] * self.res / self.i_res),
                              int(next_cell[0] * self.res / self.i_res)), self.colors["path"], thickness=4)
                    mid_y = np.linspace(current_actual[0], self.actionGrid[current_cell][0], cell_split,
                                        endpoint=False)
                    mid_x = np.linspace(current_actual[1], self.actionGrid[current_cell][1], cell_split,
                                        endpoint=False)
                    for i in range(cell_split):
                        path_points.append([mid_y[i], mid_x[i], current_theta])
                else:
                    cy, cx, cr, th1, th2, turn = self.actionGrid[current_cell][3:]
                    if th2 < th1:
                        th2 += 360
                    cv2.ellipse(self.pathImage, (int(cx), int(cy)), (int(cr), int(cr)), 0,
                                th1, th2, self.colors["path"], thickness=4)
                    mid_theta = np.linspace(th2, th1, cell_split) * pi / 180.0
                    for i in range(len(mid_theta)):
                        k = (i if turn > 0 else (len(mid_theta) - i - 1))
                        path_points.append([(cy + cr*sin(mid_theta[k])) * self.i_res,
                                            (cx + cr*cos(mid_theta[k])) * self.i_res,
                                            mid_theta[k] + pi/2.0 * turn])
                current_theta = self.actionGrid[current_cell][2]
                current_actual = self.actionGrid[current_cell][:3]
                current_cell = next_cell
                next_cell = tuple(self.parentGrid[next_cell])
            path_points.reverse()
            self.frames.append(np.copy(self.pathImage))
            self.draw_robot(self.start, self.pathImage)

        # Create array of velocities
        for i in range(len(path_points) - 1):
            py, px, pr = path_points[i]
            ny, nx, nr = path_points[i+1]
            vy = (ny - py) / self.dt
            vx = (nx - px) / self.dt
            vr = (nr - pr) / self.dt
            self.velocityArray.append([vx, vy, vr])

        # Create output video
        writer = cv2.VideoWriter('FinalAnimation.mp4', cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 30,
                                 (self.pathImage.shape[1], self.pathImage.shape[0]))
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
                self.draw_goal(self.goal, next_image)
                next_image[self.obstaclePixels] = self.colors["obstacle"]
                self.draw_robot(path_points[i], next_image)
                for j in range(1):
                    writer.write(next_image)
                    if self.play:
                        cv2.imshow(window_name, next_image)
                        cv2.waitKey(15)
            for i in range(150):
                writer.write(next_image)
            if self.play:
                cv2.imshow(window_name, next_image)
                cv2.waitKey(500)
        writer.release()
        if self.play:
            cv2.destroyWindow(window_name)

    def on_goal(self, point):
        """
        Checks to see whether the given point is on the goal (within a certain threshold).

        Parameters
        ----------
        point: tuple of float
            The point for which to check

        Returns
        -------
        bool
            True if the point is on the goal, False otherwise
        """
        return sqrt((self.goal[0] - point[0]) ** 2 + (self.goal[1] - point[1]) ** 2) <= self.goal_threshold

    def next_position(self, current_position, rpm_l, rpm_r):
        """
        Calculates the next position of a robot with a given position and wheel RPMs.

        Parameters
        ----------
        current_position: tuple of float
            Current location of the robot, in the form (y, x, theta)
        rpm_l: float
            RPM of left wheel
        rpm_r: float
            RPM of right wheel

        Returns
        -------
        tuple of float
            New location of the robot, along with distance traveled and the center of the circle of curvature,
            in the form (y, x, theta, arc_length, arc_center)

        References
        ----------
        https://robotics.stackexchange.com/questions/1653/calculate-position-of-differential-drive-robot
        """
        # Robot's current y, x, and orientation
        ry, rx, rt = current_position
        omega_l = rpm_l * 2.0 * pi / 60.0
        omega_r = rpm_r * 2.0 * pi / 60.0

        # Check for straight line first (to handle infinite radius of curvature)
        if abs(rpm_l - rpm_r) < 0.00001:
            ny = ry + self.wr * omega_r * sin(rt) * self.dt
            nx = rx + self.wr * omega_r * cos(rt) * self.dt
            nt = rt
            arc_length = self.wr * omega_r
            arc_center = -1
            turn = 0.0

        else:
            # Calculate new location for turn
            dist_l = self.wr * omega_l * self.dt
            dist_r = self.wr * omega_r * self.dt
            dist_c = (dist_l + dist_r) / 2.0
            cr = self.w_axis * (dist_r + dist_l) / (2.0 * (dist_r - dist_l))
            d_theta = (dist_r - dist_l) / self.w_axis
            ny = ry - cr * (cos(d_theta + rt) - cos(rt))
            nx = rx + cr * (sin(d_theta + rt) - sin(rt))
            nt = (rt + d_theta) % (2.0 * pi)
            arc_length = dist_c

            # Calculate arc center, for use in drawing curves
            turn = 1.0 if omega_r > omega_l else -1.0
            cy = ry + cr * sin(rt + pi / 2.0)
            cx = rx + cr * cos(rt + pi / 2.0)
            th_1 = int(atan2(ry - cy, rx - cx) * 180.0 / pi) % 360
            th_2 = int(atan2(ny - cy, nx - cx) * 180.0 / pi) % 360
            arc_center = (cy / self.i_res, cx / self.i_res, abs(cr / self.i_res),
                          th_1 if turn > 0 else th_2, th_1 if turn < 0 else th_2)
        return ny, nx, nt, arc_length, arc_center, turn

    def draw_goal(self, goal_pos, image):
        """
        Draws the robot at the indicated position.

        Parameters
        ----------
        goal_pos: tuple
            Position at which to draw the goal
        image: np.array
            Image on which to draw the goal
        """
        gy, gx = goal_pos
        rad = self.radius + self.goal_threshold
        cv2.circle(image, (int(gx / self.i_res), int(gy / self.i_res)), int(rad / self.i_res), self.colors["goal"],
                   thickness=cv2.FILLED)
        rim = 1.2
        cv2.circle(image, (int(gx / self.i_res), int(gy / self.i_res)), int(rad * rim / self.i_res),
                   self.colors["goal"])

    def draw_robot(self, robot_pos, image):
        """
        Draws the robot at the indicated position.

        Parameters
        ----------
        robot_pos: tuple
            Position at which to draw the robot
        image: np.array
            Image on which to draw the robot
        """
        ry, rx, rot = robot_pos
        rad = self.radius
        cv2.circle(image, (int(rx / self.i_res), int(ry / self.i_res)), int(rad / self.i_res), self.colors["robot"],
                   thickness=cv2.FILLED)
        cv2.circle(image, (int(rx / self.i_res), int(ry / self.i_res)), int(rad / self.i_res), (0, 0, 0),
                   thickness=(1 + int(0.01 / self.i_res)))
        rim = 1.2
        s_arrow = [[int((rx+(rad*rim)*cos(rot + pi/4)) / self.i_res),
                    int((ry+(rad*rim)*sin(rot + pi/4)) / self.i_res)],
                   [int((rx+(rad*rim)*1.4*cos(rot)) / self.i_res),
                    int((ry+(rad*rim)*1.4*sin(rot)) / self.i_res)],
                   [int((rx+(rad*rim)*cos(rot - pi/4)) / self.i_res),
                    int((ry+(rad*rim)*sin(rot - pi/4)) / self.i_res)]]
        polygon = np.array(s_arrow, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], False, self.colors["robot"], thickness=1)

    def get_velocities(self):
        return self.velocityArray


def main(argv):
    parser = argparse.ArgumentParser(
        description="Project-3 -- Phase-2:  Navigation of a rigid robot from a start point "
                    "to an end point using A* algorithm.")
    parser.add_argument('start_x', type=float, help='X-coordinate of initial node of the robot')
    parser.add_argument('start_y', type=float, help='Y-coordinate of initial node of the robot')
    parser.add_argument('start_theta', type=float, help='Start point orientation of the robot')
    parser.add_argument('goal_x', type=float, help='X-coordinate of goal node of the robot')
    parser.add_argument('goal_y', type=float, help='Y-coordinate of goal node of the robot')
    parser.add_argument('rpm1', type=float, help='Low RPM')
    parser.add_argument('rpm2', type=float, help='High RPM')
    parser.add_argument('--clearance', type=float, help='Robot clearance')
    parser.add_argument('--hw', type=float, help='Heuristic weight.  Defaults to 2.0 when omitted. '
                                                 '(Set to 1.0 for optimal path or 0.0 for Dijkstra)')
    parser.add_argument('--reduced', action="store_true",
                        help='Use a reduced set of actions (<RPM1, RPM2>; <RPM2, RPM1>; <RPM2, RPM2>)')
    parser.add_argument('--play', action="store_true", help="Play using opencv's imshow")
    args = parser.parse_args(argv)

    sx = args.start_x
    sy = args.start_y
    st = args.start_theta
    gx = args.goal_x
    gy = args.goal_y
    rpm1 = args.rpm1
    rpm2 = args.rpm2
    clearance = args.clearance
    h_weight = args.hw
    reduced = args.reduced
    p = args.play
    start_pos = (sx, sy, st)
    goal_pos = (gx, gy)
    Robot(start_pos, goal_pos, rpm1, rpm2, hw=h_weight, reduced=reduced, play=p, clearance=clearance)


if __name__ == '__main__':
    count = 0
    main(sys.argv[1:])
