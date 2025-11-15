import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import chess
from move_mapping import MoveMapper

class ChessGame:
    
    def __init__(self, include_queen_promotions=False):
       
        self.row_count = 8
        self.column_count = 8
        
        # Crear mapper
        self.move_mapper = MoveMapper(include_queen_promotions=include_queen_promotions)
        self.action_size = self.move_mapper.action_size
        
        logging.info(f" Action size: {self.action_size}")

    def get_initial_state(self):
        return chess.Board()
    
    def get_next_state(self, state, action, player):
        new_state = state.copy()
        
        # Convertir acción a movimiento
        move = self.move_mapper.action_to_move(action, new_state)
        
        if move is None:
            raise ValueError(
                f"Acción {action} no corresponde a un movimiento legal.\n"
                f"FEN: {state.fen()}"
            )
        
        if move not in new_state.legal_moves:
            raise ValueError(
                f"Movimiento {move.uci()} no es legal en esta posición.\n"
                f"FEN: {state.fen()}"
            )
        
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
  
        return self.move_mapper.get_action_mask(state)
    
    def get_legal_actions(self, state):

        return self.move_mapper.get_legal_actions(state)
    
    def get_value_and_terminated(self, state, action_taken=None):
       
        if state.is_checkmate():
            return -1, True
        
        if state.is_stalemate():
            return 0, True
        if state.is_insufficient_material():
            return 0, True
        if state.halfmove_clock >= 100:
            return 0, True
        if state.is_fivefold_repetition():
            return 0, True
        
        return 0, False
    
    def get_encoded_state(self, state):
      
        encoded = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_idx = {
            chess.PAWN: 0, 
            chess.KNIGHT: 1, 
            chess.BISHOP: 2,
            chess.ROOK: 3, 
            chess.QUEEN: 4, 
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = state.piece_at(square)
            if piece:
              
                row = 7 - (square // 8)
                col = square % 8
                
                piece_type = piece_idx[piece.piece_type]
                
                if piece.color == chess.WHITE:
                    channel = piece_type
                else:
                    channel = piece_type + 6
                
                encoded[channel, row, col] = 1
                
        return encoded
    
    def render(self, state):

        logging.info(state)
        logging.info()
    
    def get_move_from_action(self, state, action):
     
        move = self.move_mapper.action_to_move(action, state)
        
        if move is None:
            raise ValueError(
                f"Acción {action} no corresponde a un movimiento legal.\n"
                f"FEN: {state.fen()}"
            )
        
        return move
    
    def get_action_from_move(self, move):
      
        return self.move_mapper.move_to_action(move)

