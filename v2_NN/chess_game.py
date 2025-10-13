import numpy as np
import chess

class ChessGame:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 4672  
        self.move_to_index, self.index_to_move = self._create_move_mapping()
    
    def _create_move_mapping(self):
        move_to_index = {}
        index_to_move = {}
        
        idx = 0
        
        for from_square in chess.SQUARES:  # 0-63
            for to_square in chess.SQUARES:  # 0-63
                if from_square == to_square:
                    continue  
                
                move = chess.Move(from_square, to_square)
                move_uci = move.uci()
                
                # Mapear movimiento base
                move_to_index[move_uci] = idx
                index_to_move[idx] = move_uci
                idx += 1
                
                if (from_square in range(48, 56)) or (from_square in range(8, 16)):
                    # Piezas de promoción: Caballo, Alfil, Torre, Dama
                    for promotion_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                        move_promo = chess.Move(from_square, to_square, promotion=promotion_piece)
                        move_promo_uci = move_promo.uci()
                        
                        move_to_index[move_promo_uci] = idx
                        index_to_move[idx] = move_promo_uci
                        idx += 1
        
        return move_to_index, index_to_move
    
    def get_initial_state(self):

        return chess.Board()
    
    def get_next_state(self, state, action, player):
     
        move = self.get_move_from_action(state, action)
        new_state = state.copy()
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
    
        valid_moves = np.zeros(self.action_size)
        legal_moves = list(state.legal_moves)
        
        # ⭐ USAR MAPEO GLOBAL: buscar índice del movimiento
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci in self.move_to_index:
                idx = self.move_to_index[move_uci]
                valid_moves[idx] = 1
        
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
        """Retorna el oponente del jugador"""
        return -player
    
    def get_opponent_value(self, value):
        """Invierte el valor desde la perspectiva del oponente"""
        return -value
    
    def change_perspective(self, state, player):
    
        return state
    
    def get_encoded_state(self, state):
      
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
        print(state)
        print()
    
    def get_move_from_action(self, state, action):
     
        if action >= len(self.index_to_move) or action < 0:
            raise ValueError(f"Acción {action} fuera de rango (0-{len(self.index_to_move)-1})")
        
        # Obtener el movimiento UCI del mapeo global
        move_uci = self.index_to_move[action]
        
        # Convertir a objeto chess.Move
        try:
            move = chess.Move.from_uci(move_uci)
        except:
            raise ValueError(f"No se pudo parsear movimiento: {move_uci}")
        
        # Verificar que es legal (seguridad)
        if move not in state.legal_moves:
            raise ValueError(
                f"Movimiento {move_uci} (índice {action}) no es legal en esta posición.\n"
                f"Movimientos legales: {[m.uci() for m in list(state.legal_moves)[:5]]}..."
            )
        
        return move