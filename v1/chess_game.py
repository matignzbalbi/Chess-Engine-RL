import numpy as np
import chess

## Definimos la clase Ajedrez y sus atributos.

class ChessGame:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        # Para simplificar, usaremos un mapeo de movimientos a índices
        # En una implementación completa, necesitaríamos los 4672 movimientos posibles
        # Por ahora, usaremos el índice de la lista de movimientos legales
        self.action_size = 4672  # Como en AlphaZero
        
    def get_initial_state(self):
        """Retorna el estado inicial del tablero"""
        return chess.Board()
    
    def get_next_state(self, state, action, player):
        """
        Aplica una acción al estado y retorna el nuevo estado.
        
        Args:
            state: chess.Board actual
            action: índice del movimiento en la lista de movimientos legales
            player: 1 para blancas, -1 para negras (no usado directamente por python-chess)
            
        Returns:
            Nuevo estado después de aplicar la acción
        """
        new_state = state.copy()
        legal_moves = list(new_state.legal_moves)
        move = legal_moves[action]
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
        """
        Retorna un vector binario indicando movimientos válidos.
        
        Args:
            state: chess.Board actual
            
        Returns:
            Array numpy binario donde 1 = movimiento válido
        """
        valid_moves = np.zeros(self.action_size)
        legal_moves = list(state.legal_moves)
        
        # Marcar los primeros N movimientos como válidos
        # (simplificación: usamos índices secuenciales)
        for i in range(len(legal_moves)):
            valid_moves[i] = 1
            
        return valid_moves
    
    def check_win(self, state):
        """
        Verifica si el juego terminó y quién ganó.
        
        Returns:
            1 si ganan blancas, -1 si ganan negras, 0 si empate/continúa
        """
        if state.is_checkmate():
            # El jugador actual perdió (está en jaque mate)
            return -1 if state.turn == chess.WHITE else 1
        return 0
    
    def get_value_and_terminated(self, state, action_taken):
        """
        Retorna el valor del estado y si es terminal.
        
        Returns:
            (value, is_terminal): tupla con valor y booleano
        """
        if state.is_checkmate():
            # El jugador actual (cuyo turno es) está en jaque mate
            value = -1  # Perdió el jugador actual
            return value, True
        
        if state.is_stalemate() or state.is_insufficient_material() or \
           state.is_seventyfive_moves() or state.is_fivefold_repetition():
            return 0, True  # Empate
        
        return 0, False  # Juego continúa
    
    def get_opponent(self, player):
        """Retorna el oponente del jugador"""
        return -player
    
    def get_opponent_value(self, value):
        """Invierte el valor desde la perspectiva del oponente"""
        return -value
    
    def change_perspective(self, state, player):
        """
        Cambia la perspectiva del tablero al jugador dado.
        En ajedrez con python-chess, el estado ya maneja esto internamente.
        """
        return state
    
    def get_encoded_state(self, state):
        """
        Codifica el estado del tablero como un array numpy.
        Esto será útil cuando agregues la red neuronal.
        
        Returns:
            Array numpy representando el tablero
        """
        # Representación simple: 12 canales (6 piezas x 2 colores)
        # + información adicional como en AlphaZero
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
        """Imprime el tablero de forma legible"""
        print(state)
        print()
    
    def get_move_from_action(self, state, action):
        """Convierte un índice de acción en un movimiento de ajedrez"""
        legal_moves = list(state.legal_moves)
        return legal_moves[action]