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
    # end, player_score, opponent_score = check_endgame(chess_board, player, opponent)
    # if end:
    #   return self.evaluate_board(chess_board, player, player_score, opponent_score), None
    
    # moves = get_valid_moves(chess_board, player)
    
    # if not moves:
    #   return None
    
    # initial_alpha = float('-inf')
    # initial_beta = float('inf')
    # best = None
    
    # for move in moves_AI:
    #   #Make a copy of the chessboard since might have to compute a lot of moves
    #   simulated = deepcopy(chess_board)
    #   execute_move(simulated, move, player)
    #   _, player_score, opponent_score = check_endgame(simulated, player, opponent)
    #   alpha = self.evaluate_board(simulated, player, player_score, opponent_score, player)

    #   if alpha > initial_alpha:
    #     initial_alpha = alpha
    #     best = move
    # legal_moves = get_valid_moves(chess_board, player)

    # if not legal_moves:
    #     return None  # No valid moves available, pass turn

    # # Advanced heuristic: prioritize corners and maximize flips while minimizing opponent's potential moves
    # best_move = None
    # best_score = float('-inf')

    # for move in legal_moves:
    #     simulated_board = deepcopy(chess_board)
    #     execute_move(simulated_board, move, player)
    #     _, player_score, opponent_score = check_endgame(simulated_board, player, opponent)
    #     move_score = self.evaluate_board(simulated_board, player, player_score, opponent_score)

    #     if move_score > best_score:
    #         best_score = move_score
    #         best_move = move

    #     # Return the best move found
       

    #   if initial_beta <= initial_alpha:
    #     break

    # max_time = 2
    depth = 10  # Define the search depth

    best_move = self.iterative_deepening(chess_board, player, opponent, depth)
    
    # Return the move that maximizes the AI's advantage
    #return best_move
  

    # alpha = float('-inf')
    # beta = float('inf')

    # best_move = None
    # best_score = float('inf')  # Start with a large value for minimizing

    # valid_moves = get_valid_moves(chess_board, player)
    # if not valid_moves:
    #     return None  # Pass turn if no valid moves

    # for move in valid_moves:
    #     simulated_board = deepcopy(chess_board)
    #     execute_move(simulated_board, move, player)

    #     # Run alpha-beta pruning for opponent (maximizing player)
    #     score, _ = self.alpha_beta_pruning(simulated_board, player, opponent, depth, alpha, beta, True)

    #     if score < best_score:  # Minimize the opponent's maximum score
    #         best_score = score
    #         best_move = move

    #     beta = min(beta, score)
    #     if beta <= alpha:
    #         break  # Alpha cutoff

  

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    #return best
    return best_move 
  
  def iterative_deepening(self, board, player, opponent, max_depth, max_time=2):
    """
    Perform iterative deepening with alpha-beta pruning for AI moves.
    """
    best_move = None
    depth = 0
    start_time = time.time()
    
    # Run iterative deepening up to the maximum allowed time
    while time.time() - start_time <= max_time and depth <= max_depth:
        move = self.alpha_beta_pruning(board, player, opponent, max_depth, float('-inf'))
        depth += 1
    
    if move is not None:
      best_move = move
        
    return best_move
  
  def alpha_beta_pruning(self, board, player, opponent, depth, alpha):
    """
    Perform alpha-beta pruning for Othello with the AI opponent as the maximizing player.

    Parameters:
    - board: 2D numpy array representing the game board.
    - player: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
    - opponent: Integer representing the opponent's color.
    - depth: Current depth of the search tree.
    - alpha: Alpha value for pruning.
    - beta: Beta value for pruning.
    - maximizing_player: Boolean (always True for the opponent in this setup).
    - evaluate_func: Evaluation function for board state.

    Returns:
    - tuple: (best_score, best_move)
    """
    # Check for game over or depth limit
    end, _, _ = check_endgame(board, player, opponent)
    if end or depth == 0:
        return self.evaluate_board(board, player, player_score, opponent_score)

    valid_moves = get_valid_moves(board, player)
    if not valid_moves:  # Pass turn if no valid moves
        return None
    
    best_move = None
    max_eval = float('-inf')
    beta = float('inf')
    for move in valid_moves:
        simulated_board = deepcopy(board)
        execute_move(simulated_board, move, player)
        opp_valid_moves = get_valid_moves(simulated_board, opponent)
        if opp_valid_moves:
            opponent_avg_score = 0
            for opp_move in opp_valid_moves:
              simulated_board2 = deepcopy(simulated_board)
              execute_move(simulated_board2, opp_move, opponent)
              _, player_score, opponent_score = check_endgame(simulated_board, player, opponent)
              child_value = self.evaluate_board(simulated_board, player, player_score, opponent_score)
              opponent_avg_score += child_value
            opponent_avg_score /= len(opp_valid_moves)
        # else:
        #     # Opponent has no moves
        #     opponent_avg_score = self.alpha_beta_pruning(simulated_board2, player, opponent, depth - 1, alpha, beta)
        _, player_score, opponent_score = check_endgame(simulated_board, player, opponent)
        eval_score = self.evaluate_board(simulated_board, player, player_score, opponent_score)
        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move
        alpha = max(alpha, eval_score)
        if beta <= alpha:
            break  # Beta cutoff
    return max_eval, best_move if best_move else random_move(board, player)
  
  def evaluate_board(self, board, player, player_score, opponent_score):
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
        corner_score = sum(1 for corner in corners if board[corner] == player) * 25
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - player) * -25

        # Mobility: the number of moves the opponent can make
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, 3 - player))
        mobility_score = player_moves - opponent_moves

        x_squares = [(1, 1), (1, board.shape[1] - 2), (board.shape[0] - 2, 1), (board.shape[0] - 2, board.shape[1] - 2)]
        c_positions = [(0, 1), (0, board.shape[1] - 2), (1, 0), (1, board.shape[1] - 1),
        (board.shape[1] - 2, 0), (board.shape[1] - 2, board.shape[1] - 1), 
        (board.shape[1] - 1, 1), (board.shape[1] - 1, board.shape[1] - 2)]

        num_moves_played = np.count_nonzero(board)  # Count the number of non-zero (occupied) cells
        early_game_threshold = 20  # Change this based on game observation
        x_score = 0
        c_score = 0
        x_penalty = 0
        c_penalty = 0

        if num_moves_played < early_game_threshold:
          # X-squares: Positions adjacent to corners (high negative weight early)
          # C-square heuristic (avoid early)
          x_score = -15 * sum(1 for pos in x_squares if board[pos] == player)
          c_score = -10 * sum(1 for pos in c_positions if board[pos] == player)
          x_penalty = 15 * sum(1 for pos in x_squares if board[pos] == 3 - player)
          c_penalty = 10 * sum(1 for pos in c_positions if board[pos] == 3 - player)
        else:
          x_score = -5 * sum(1 for pos in x_squares if board[pos] == player)
          c_score = -3 * sum(1 for pos in c_positions if board[pos] == player)
          x_penalty = 5 * sum(1 for pos in x_squares if board[pos] == 3 - player)
          c_penalty = 3 * sum(1 for pos in c_positions if board[pos] == 3 - player)

        # Stability (stable discs) and edge stability
        stable_score = 0
        # Simple idea: assume discs at edges and corners are more stable
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if r == 0 or r == board.shape[0] - 1 or c == 0 or c == board.shape[1] - 1:
                    if board[r, c] == player:
                        stable_score += 5
                    elif board[r, c] == 3 - player:
                        stable_score -= 5

        # Combine scores
        total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score + x_score + x_penalty + c_score + c_penalty + stable_score
        return total_score
  
  




  
  

