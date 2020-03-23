# ENPM 661 - A* Implementation on Rigid Robot - Project 3 (Phase-2)  - Group 8

### Description

This program uses A* algorithm to find the optimal path
in a given map for a Rigid robot.
The user can provide an initial state and a goal state,
and the algorithm will output visualization of the 
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

Additional file, obstacleMap.py is utilized when the above execution happens.
obstacleMap.py - The file creates a map of obstacles which are generated using half-planes in a continuous space.
Collisions are handled in the obstacleMap.py file by checking a few calculations for straight polygons:
(1) Checking the distance to each edge
(2) Checking the distance to each vertex
(3) Checking to see if the angle between two adjacent vertices and the point in question in clockwise order is ever
negative (using the range -pi to pi).  The circle and ellipse simply use the distance from the center to the point.

The main file consists of a class from which an object created would have an instance of the main algorithms
such A* algorithm, checks for the cells in obstacles and outer map, 
backtracking required for the initial node to reach the goal node.

If your python command is different, for example python3, adjust accordingly.
This script requires python 3.5+ to run.

--------------------------------------

### Arguments
 
The program takes the x,y - coordinates of the initial and goal cells separately as the arguments and also
theta value for the initial cell.
In addition to these arguments it needs radius, clearance and step-size (which is the distance to advance every step).
Optionally, a theta for the goal cell may be specified at the end.
Also, to change the value of the weight for the heuristic, use `--hw value`.  For example,
use `--hw 3.0` to multiply the heuristic function (which is the Euclidean distance) by 3.0.
WARNING: setting the heuristic to < 1.2 is much slower.  Default value is 2.0 when omitted.

Additional help with using the program can be found by running the command `python Astar_rigid.py --help`.

NOTE:  This program outputs an mp4 file with the name FinalAnimation.mp4 in the working directory.
If generating this mp4 file does not work, alternatively you can append the flag `--play` at the end
of the command to use openCV's built-in imshow command.
The mp4 file may still work even if there is an error message in the console related to FFmpeg.
Preferably omit the `--play` flag if the mp4 file is playable.
Required arguments are in the following order:
        
        Ix Iy theta Gx Gy R C S

Some examples of valid commands (the top one used for the deliverable):

        python Astar_rigid.py 50 30 64 150 150 1 1 1
        
        python Astar_rigid.py 10 10 290 190 150 2 1 3 --hw 1.5
        
        python Astar_rigid.py 50 30 64 150 150 1 1 1 --theta_g 240 --play

--------------------------------------


### Output

While the program is running, it will output the visualization of the cells being explored.
Once a solution has been found, the program outputs the cells explored using the using the vectors.
The program also outputs an optimal sequence of actions to get
from the initial to the goal point.
