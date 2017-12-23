#!/usr/bin/env python
# nrooks.py : Solve the N-Rooks problem!
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

import sys

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] )

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] )

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ " ".join([ "R" if col else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state (initial successor function)
def successors(board):
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]

# Get list of successors of given board state (Modified successor function)
def successors2(board,fringe):
    r= count_pieces(board)
    if r >= N :
        return successors2( fringe.pop(), fringe)
    all_successor = [add_piece(board, r, c) for c in range(0,N)]
    valid_successor=[]
    for i in range(len(all_successor)):
        if all( [ count_on_row(all_successor[i], r) <= 1 for r in range(0, N) ] ) and \
        all( [ count_on_col(all_successor[i], c) <= 1 for c in range(0, N) ] ):
           valid_successor.append(all_successor[i])
    if len(valid_successor) == 0 :
          return successors2( fringe.pop(), fringe)
    return valid_successor


# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors2( fringe.pop(), fringe ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[1])

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N]*N
solution = solve(initial_board)
print (printable_board(solution) if solution else "Sorry, no solution found. :(")
