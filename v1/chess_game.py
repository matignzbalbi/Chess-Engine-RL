import numpy as np
import chess

## Definimos la clase Ajedrez y sus atributos.

class ChessGame:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 4672  
        
    def get_initial_state(self):
        return chess.Board()
    
    def get_next_state(self, state, action, player):
        
        new_state = state.copy()
        legal_moves = list(new_state.legal_moves)
        move = legal_moves[action]
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
        
        valid_moves = np.zeros(self.action_size)
        legal_moves = list(state.legal_moves)
        
        # Marcar los primeros N movimientos como válidos
        for i in range(len(legal_moves)):
            valid_moves[i] = 1
            
        return valid_moves
    
    def check_win(self, state):
       
        if state.is_checkmate():
            # El jugador actual perdió (está en jaque mate)
            return -1 if state.turn == chess.WHITE else 1
        return 0
    
    def get_value_and_terminated(self, state, action_taken):
     
        if state.is_checkmate():
            # El jugador actual (cuyo turno es) está en jaque mate
            value = -1  # Perdió el jugador actual
            return value, True
        
        if state.is_stalemate() or state.is_insufficient_material() or \
           state.is_seventyfive_moves() or state.is_fivefold_repetition():
            return 0, True  # Empate
        
        return 0, False 
    
    def get_opponent(self, player):
    
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):

        return state
    
    def get_encoded_state(self, state):
    
        # 12 canales (6 piezas x 2 colores)

        encoded = np.zeros((8, 8, 12))
        
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = state.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                piece_type = piece_idx[piece.piece_type]
                channel = piece_type if piece.color == chess.WHITE else piece_type + 6
                encoded[row, col, channel] = 1
                
        return encoded
    
    def render(self, state):
        print(state)
        print()
    
    def get_move_from_action(self, state, action):
      
        legal_moves = list(state.legal_moves)
        return legal_moves[action]