# ENPM 661 - A* Implementation on Rigid Robot - Project 3(Phase-2)  - Group 8

### Description

This program uses A* algorithm to find the optimal path
in a given map for a Rigid robot.
The user can provide an initial state and a goal state,
and the algorithm will output vizualization of the 
cells explored and the optimal path in reaching the goal.

--------------------------------------

### Libraries

The solver uses the following Python libraries:

| Name      | Usage                                                             |
| --------- | ----------------------------------------------------------------- | 
| cv2       | OpenCV for vizualization                                          |
| numpy     | For scientific computing                                          |
| sys       | Outputting to stdout                                              |
| math      | To use the mathematical functions                                 |
| argparse  | Parsing input arguments to the program                            |
| datetime  | To get the runtime of the project		                            	|

--------------------------------------

### Execution and Explanation
The program is run by executing `python Astar_rigid.py`

Additional file, obstacleMap.py is run when the above execution happens.
obstacleMap.py - The file creates a map of obstacles which are generated using half-planes in a continuous space.

The main file consists of a class from which an object created would have an instance of the main algorithms
such A* algorithm, checks for the cells in obstacles and outer map, 
backtracking required for the initial node to reach the goal node.

If your python command is different, for example python3, adjust accordingly.
This script requires python 3.x to run.

--------------------------------------

### Arguments
 
The program takes the x,y - coordinates of the initial and goal cells seperately as the arguments and also theta value for the initial cell.
Inaddition to these it needs radius, clearance and step-size.

Help with using the program can be found by running the command `python solver.py --help`.

NOTE:  This program outputs an mp4 file.
If generating this mp4 file does not work, alternatively you can append the flag `--play` at the end
of the command to use openCV's built-in imshow command.
The mp4 file may still work even if there is an error message in the console related to FFmpeg.
Preferably omit the `--play` flag if the mp4 file is playable.

Some examples of valid commands:

        python Astar_rigid.py 50 30 30 150 150 1 1 1 (arg : Ix Iy theta Gx Gy R C S)

--------------------------------------


### Output

While the program is running, it will output the vizualization of the cells being explored.
Once a solution has been found, the program outputs the cells explored using the using the vectors.
The program also outputs an optimal sequence of actions to get
from the initial to the goal point.
