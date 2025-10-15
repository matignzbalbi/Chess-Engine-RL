import numpy as np
import chess

class ChessGame:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 4672  # Como en AlphaZero
    
    def get_initial_state(self):
        """Retorna el estado inicial del tablero"""
        return chess.Board()
    
    def get_next_state(self, state, action, player):

        move = self.get_move_from_action(state, action)
        new_state = state.copy()
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
     
        valid_moves = np.zeros(self.action_size)
        legal_moves = list(state.legal_moves)
        
        # Marcar los primeros N movimientos como válidos
        # (N = número de movimientos legales en esta posición)
        for i in range(len(legal_moves)):
            if i < self.action_size:
                valid_moves[i] = 1
        
        return valid_moves
    
    def check_win(self, state):
   
        if state.is_checkmate():
            # El jugador actual (cuyo turno es) está en jaque mate
            return -1 if state.turn == chess.WHITE else 1
        return 0
    
    def get_value_and_terminated(self, state, action_taken):
   
        if state.is_checkmate():
            # El jugador actual (cuyo turno es) está en jaque mate
            value = -1  # Perdió el jugador actual
            return value, True
        
        if state.is_stalemate() or state.is_insufficient_material() or \
           state.is_seventyfive_moves() or state.is_repetition(3):
            return 0, True  # Empate
        
        return 0, False  # Juego continúa
    
    def get_opponent(self, player):
    
        return -player
    
    def get_opponent_value(self, value):
 
        return -value
    
    def change_perspective(self, state, player):
    
        return state
    
    def get_encoded_state(self, state):

        # Formato: (canales, filas, columnas) para PyTorch
        encoded = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = state.piece_at(square)
            if piece:
                row = 7 - (square // 8)  # Invertir fila (chess usa 0=a1, queremos 0=a8)
                col = square % 8
                piece_type = piece_idx[piece.piece_type]
                
                # Canal depende del color
                if piece.color == chess.WHITE:
                    channel = piece_type
                else:
                    channel = piece_type + 6
                
                encoded[channel, row, col] = 1
                
        return encoded
    
    def render(self, state):
        board = state.board if hasattr(state, "board") else state

        print("\n  a  b  c  d  e  f  g  h")
        print("  +-----------------+")

        for rank in range(8, 0, -1):
            row = []
            for file in range(8):
                square = chess.square(file, rank - 1)
                piece = board.piece_at(square)
                if piece:
                    symbol = piece.unicode_symbol(invert_color=True)
                else:
                    symbol = "·"
                row.append(symbol)
            print(f"{rank} | {' '.join(row)} |")
        print("  +-----------------+")
        print("  a  b  c  d  e  f  g  h\n")
    
    def get_move_from_action(self, state, action):
 
        legal_moves = list(state.legal_moves)
        
        if action >= len(legal_moves) or action < 0:
            raise IndexError(
                f"Acción {action} fuera de rango. "
                f"Movimientos legales: {len(legal_moves)}"
            )
        
        return legal_moves[action]