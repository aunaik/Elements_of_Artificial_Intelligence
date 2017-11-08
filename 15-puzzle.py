#!/usr/bin/env python
# B551 Elements of AI, Prof. David Crandall
# Created by: Akshay Naik, 2017
# The 15-puzzle problem: Given any 4X4 board configuration, the goal state should be reached by sliding one, two or three tiles
# if it is reachable from the initial state.

#Problem abstraction
# State: A state consist of a list of total cost[f(x)], heuristic value [h(x)], cost of path till current state [g(x)],
#        path from initial state to current state and the puzzle board.
# Set of states (S): Set of 4X4 boards wherein 1-15 numbers and 0 arranged are arbitarily.
#                    Where only those boards which can lead to goal state are considered
#                    Therefore, n(s)= 16!/2
# Initial state (I): Any board configuration having 1-15,0 numbers arranged on it.
# Successor function: Successor function generally returns a list of six successors for the given state of board by sliding one,two or three
#                     tiles vertically or horizontally.
# Goal state (G): Here, the goal state is 1-15,0 numbers arranged in order on 4X4 board.
# Cost function: Assuming the cost of moving from one state to another state as 1(Uniform cost function).
#                i.e. C(Sa,Sb)=1, where Sb is successor state of state Sa
#                The path cost is the number of state transitions required to reach goal state from initial state.
# Heuristic Function: Heuristic used here is Linear conflict which considers manhattan distance to move each tile from its current position to
#                     goal state position and divide the total manhattan distance by 3 to make the heuristic function admissible and also penalize
#                     with +2 whenever there is linear conflict between two tiles
#                     Linear conflict: Whenever there are two tiles with value i and j placed in same row or column as goal state
#                                      such that tile with vaule i is placed before tile with value j in current state but is placed
#                                      in opposite order in goal state then we consider it as a linear conflict

# The heuristic used here is both admissible and consistent as it never overestimates at any given state(as we have divided the total manhattan distance
# by 3 to cater sliding of three tiles in a single move) and also the triangular inequality is satisfied at every state.

# Algorithm design facets
# We have used A* algorithm to solve this problem with some variations.
# Firstly, instead of checking for closed state after generation of a successor, the algorithm checks for closed state after popping out a state from priority queue.
# This reduced the number of searches in the closed state list, which inturn reduced the execution time without affecting the optimality of the solution path.
# Secondly, We have removed the state check in the fringe to find out whether the currently generated successor board configuration is already present in the fringe,
# to keep the minimum cost instance of state in fringe. This did not affect the solution optimality, but drastically reduced the execution time as popping a state and
# again ordering the states in fringe takes alot of time to execute.(Credit goes to Piazza discussion, where professor D. Crandall mentioned that sorting heapqueue in
# python is very expensive process).
# The rest of the algorithm is same as A* implementation.

# Difficluties faced and observations
# Initially we used the linear conflict algorithm by referring "https://academiccommons.columbia.edu/catalog/ac:141289". This technical paper explains actual concept
# of linear conflict and proposes an algorithm to the find linear conflict. But after implementing this algorithm, the execution time increased as the linear conflict
# calcuation was taking alot to time. So we tried to find an alternate way to calculate linear conflict which won't take much computation time. We refered a java code to
# understand the logic and have sited it above the linear conflict function in this code.
# We observed that searching the closed states list to find whether a state is closed or not takes more computation time as compared to calculating the heuristic,
# so removing the check for each successor and checking for closed state after priority popping reduces execution time drastically as number of closed state list check
# reduces by a factor of 6.
# Similarly, not checking the fringe for similar state reduces the program exectuion time.

import sys
from copy import deepcopy
from operator import itemgetter
import heapq
import math

# Created a reference dictionary to track the position of each tile in the goal state to calculate Manhattan distance
ref = {1:[0,0],2:[0,1],3:[0,2],4:[0,3],5:[1,0],6:[1,1],7:[1,2],8:[1,3],9:[2,0],10:[2,1],11:[2,2],12:[2,3],13:[3,0],14:[3,1],15:[3,2]}
# List to track the states already visited
closed=[]
fringe = []
# list to store the moves we take to reach goal state from initial configuration of the 15-puzzle
path=[]
# list to store the 15-puzzle configuration
puzzle = []

# Check whether the given state is a goal state or not
def is_goal(state):
    if state[1]==0:
        return True
    return False

# Printing the output
def print_puzzle(path):
     for i in range(len(path)):
         print path[i],

# Check whether the current state is visited earlier or not
def isNotVisited(puzzle):
    for k in range(len(closed)):
        if closed[k]==puzzle:
            return False
    return True

# Calculates Linear conflict
# Discussed linear conflict logic with Siddharth Pathak and referred "https://github.com/jDramaix/SlidingPuzzle/blob/master/src/be/dramaix/ai/slidingpuzzle/server/search/heuristic/LinearConflict.java",
# which is a java code to understand how linear conflict logic works. In the code referred above there was a small bug, the algorithm did not
# consider linear conflict in the last column which is taken care in the following code.
def linear_conflict(puzzle):
    LC=0
    for i in range(4):
        max_row_value=-1
        max_col_value=-1
        for j in range(4):
            if puzzle[i][j]!=0 and int((puzzle[i][j]-1)/4) == i:
                if puzzle[i][j]>max_row_value:
                    max_row_value=puzzle[i][j]
                else:
                    LC+=2
            if puzzle[j][i]!=0 and int((puzzle[j][i])%4) == (i+1)%4:
                if puzzle[j][i]>max_col_value:
                    max_col_value=puzzle[j][i]
                else:
                    LC+=2
    return LC

# Calculates heuristic value for a state using Manhattan distance and calls linear_conflict function
# and returns total heuristic as sum of manhattan distance and linear conflicts
def heuristic(puzzle):
    mhd=0
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] in ref:
                x=ref.get(puzzle[i][j])
                mhd+=abs(i-x[0])+abs(j-x[1])
    return int(math.ceil(mhd/3.0)) + linear_conflict(puzzle)

# Successor function generates six successors for each state and add those successors to the firnge
def successors(a):
    try:
        global count
        row=col=0
        breaker =False
        for row in range(len(a[4])):
            for col in range(len(a[4])):
                if a[4][row][col] == 0:
                    breaker = True
                    break
            if breaker == True:
                break
        b=deepcopy(a)
        ex = row
        temp1=[a[4][row][i] for i in range(len(a[4]))]
        temp2=[row[col] for row in a[4]]
        temp1.pop(col)
        temp2.pop(ex)
        for i in range(len(temp1)+1):
            if i!=col:
                c = deepcopy(a)
                rowRotatedList =temp1[0:i] + [0] + temp1[i:len(temp1)]
                c[4][ex]=rowRotatedList
                h=heuristic(c[4])
                g=c[2]+1
                x=h+g
                if (i<col):
                    c[3].append("".join(["R"] + [str(col-i)] + [str(ex+1)]))
                if (i>col):
                    c[3].append("".join(["L"] + [str(i-col)] + [str(ex+1)]))
                heapq.heappush(fringe,[x,h,g,c[3],c[4]])
        for j in range(len(temp2)+1):
            if j!=ex:
                colRotatedList = temp2[0:j] + [0] + temp2[j:len(temp2)]
                c=deepcopy(a)
                for i in range(len(colRotatedList)):
                    c[4][i][col]=colRotatedList[i]
                h=heuristic(c[4])
                g=c[2]+1
                x=h+g
                if (j<ex):
                    c[3].append("".join(["D"] + [str(ex-j)] + [str(col+1)]))
                if (j>ex):
                    c[3].append("".join(["U"] + [str(j-ex)] + [str(col+1)]))
                heapq.heappush(fringe,[x,h,g,c[3],c[4]])
    except IndexError:
        print "Input format not proper"
        exit()

# Solves 15-puzzle problem using A* Algorithm
def solve(initial_state):
    heapq.heappush(fringe,initial_state)
    while len(fringe) > 0:
        s=heapq.heappop(fringe)
        if isNotVisited(s[4]):
            closed.append(s[4])
            if is_goal(s):
                return(s)
            successors(s)

# Read the initial board configuration from input file
with open(sys.argv[1],'r') as file:
    for line in file:
        puzzle.append(map(int,line.split()))

# Converting list of list to list
a= [puzzle[row][col] for row in range(4) for col in range(4)]
parity=0
zero_pos=-1
# Calculating parity of the initial board and finding out the row number of 0
for i in range(len(a)):
    for j in range(i,len(a)):
        if a[i] == 0:
            zero_pos = int(round(i/4))
            break
        elif a[j] == 0:
            continue
        elif a[i]>a[j]:
            parity+=1

# Checking whether the gievn board is solvable or not
if int((parity+zero_pos)%2) == 0:
    print "Solution not possible for the given puzzle"
    exit()

#calcuate heuristic value of initial board
h=heuristic(puzzle)
# Checks whether the initial_state is the goal stateor not
if(heuristic(puzzle)==0):
    print "The initial state itself is the goal state"
    exit()
solution=solve([h,h,0,path,puzzle])
print_puzzle(solution[3])
