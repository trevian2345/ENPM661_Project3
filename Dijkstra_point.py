import argparse
from baseRobot import Robot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Project 2:  Navigation of a point robot from a start point to an"
                                                 "end point.")
    parser.add_argument('initialX', type=int, help='X-coordinate of initial node of the robot')
    parser.add_argument('initialY', type=int, help='Y-coordinate of initial node of the robot')
    parser.add_argument('goalX', type=int, help='X-coordinate of goal node of the robot')
    parser.add_argument('goalY', type=int, help='Y-coordinate of goal node of the robot')
    parser.add_argument('--play', action="store_true", help="Play using opencv's imshow")
    args = parser.parse_args()

    sx = args.initialX
    sy = args.initialY
    gx = args.goalX
    gy = args.goalY
    p = args.play
    start_pos = (sx, sy)
    goal_pos = (gx, gy)
    pointRobot = Robot(start_pos, goal_pos, 0, 0, rigid=False, play=p)
