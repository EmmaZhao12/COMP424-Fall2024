# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    #Checks if game is over, if it is return the score, score can be calculated in a separate function
    end, player_score, opponent_score = check_endgame(chess_board, player, opponent)
    if end:
      return None
    
    moves_AI = get_valid_moves(chess_board, player)
    moves_opponent = get_valid_moves(chess_board, opponent)
    
    if not moves_AI:
      return None
    
    initial_alpha = float('-inf')
    initial_beta = float('inf')
    best = None
    
    for move in moves_AI:
      #Make a copy of the chessboard since might have to compute a lot of moves
      simulated = copy.deepcopy(chess_board)
      execute_move(simulated, move, player)
      _, player_score, opponent_score = check_endgame(simulated, player, opponent)
      alpha = self.evaluate_board(simulated, player, player_score, opponent_score)

      if alpha > initial_alpha:
        initial_alpha = alpha
        best = move

      if initial_beta <= initial_alpha:
        break

    for move in moves_opponent:
      #Make a copy of the chessboard since might have to compute a lot of moves
      simulated = copy.deepcopy(chess_board)
      execute_move(simulated, move, player)
      _, player_score, opponent_score = check_endgame(simulated, player, opponent)
      beta = self.evaluate_board(simulated, opponent, player_score, opponent_score)

      if beta < initial_beta:
        initial_beta = beta
        best = move

      if initial_beta <= initial_alpha:
        break

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return best
  
  def evaluate_board(self, board, color, player_score, opponent_score):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -10

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - color))
        mobility_score = -opponent_moves

        # Combine scores
        total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score
        return total_score

