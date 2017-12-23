#!/usr/bin/env python
# nrooks.py : Solve the N-Rooks problem!
# The N-queen problem is: Given an empty NxN chessboard, place N queens on the board so that no queens
# can take any other, i.e. such that no two queens share the same row or column or diagonal.

import sys

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] )

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] )

# Checking whether the queen is placed on unavailabe position on board
def is_valid_pos(board):
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 1 and i == u[0] and j == u[1]:
                return False
    return True

# Check the diagonal condition.
# Referred https://www.youtube.com/watch?v=p4_QnaTIxkQ for the logic of diagonal check which is been modified to some extend and incorporated here.
def diag_check(board):
    queen_pos = [[0 for j in range(0,count_pieces(board))] for i in range(0,2)]
    c=0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 1:
                queen_pos[0][c] = i
                queen_pos[1][c] = j
                c+=1
    if count_pieces(board) > 1:
        for i in range(0,count_pieces(board)-1):
            for j in range(i+1,count_pieces(board)):
                if abs(queen_pos[0][i]-queen_pos[0][j]) == abs(queen_pos[1][i]-queen_pos[1][j]):
                    return False
    return True


# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    if piece_type == 'nqueen':
        f = "Q"
    else:
         f = "R"
    q = "\n".join([ " ".join([ f if col else "X" if u==(i,j) else "_" for j,col in enumerate(row)]) for i,row in enumerate(board)])
    return q

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state
def successors(board,fringe):
    r= count_pieces(board)
    if r > N :
        return successors( fringe.pop(), fringe)
    valid_successor = [add_piece(board, r, c) for c in range(0,N)]
    final_successor=[]
    for i in range(len(valid_successor)):
        if all( [ count_on_row(valid_successor[i], r) <= 1 for r in range(0, N) ] ) and \
           all( [ count_on_col(valid_successor[i], c) <= 1 for c in range(0, N) ] ) and \
        is_valid_pos(valid_successor[i]):
            if piece_type == 'nqueen':
                if diag_check(valid_successor[i]):
                    final_successor.append(valid_successor[i])
            elif piece_type == 'nrook':
                final_successor.append(valid_successor[i])
            else:
                print   "Incorrect piece name\n" "It should be either 'nqueen' or 'nrook'"
                exit()
    if len(final_successor) == 0:
          return successors( fringe.pop(), fringe)
    return final_successor

# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors( fringe.pop(), fringe ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

#Type of piece, which determines whether the problem is nqueens or nrooks
piece_type = sys.argv[1]

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[2])

# Checking whether solution is possible or not for given number of queens
if piece_type == 'nqueen' and N < 4 and N > 1:
    print "\nSorry...No solution found for N: ", N
    exit()

# Index of unavailable position on board
u = (int(sys.argv[3])-1,int(sys.argv[4])-1)

#Check whether correct index is inputted for unavailable position
if (u[0] < -1 or u[1] <-1) or (u[0] > N-1 or u[1] > N-1):
    print "\n\nIncorrect co-ordinates are selected for the unavailable position on board\n\n"
    exit()

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N]*N
solution = solve(initial_board)
print printable_board(solution)
