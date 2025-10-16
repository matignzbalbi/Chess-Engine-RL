import numpy as np
import chess
from move_mapping import MoveMapper

class ChessGame:
    """
    Interfaz del juego de ajedrez con mapeo global de movimientos.
    
    CAMBIO CRÍTICO: Ahora usa MoveMapper para tener un mapeo consistente
    entre índices de acción y movimientos, independiente de la posición.
    """
    
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 4672  # Como en AlphaZero
        
        # Sistema de mapeo global
        self.move_mapper = MoveMapper()
        
        print("✓ ChessGame inicializado con mapeo global de movimientos")
        
    def get_initial_state(self):
        """Retorna el estado inicial del tablero"""
        return chess.Board()
    
    def get_next_state(self, state, action, player):
        """
        Aplica una acción al estado y retorna el nuevo estado.
        
        Args:
            state: chess.Board actual
            action: índice de acción global (0-4671)
            player: 1 para blancas, -1 para negras (no usado por python-chess)
            
        Returns:
            Nuevo estado después de aplicar la acción
        """
        new_state = state.copy()
        
        # Convertir acción a movimiento usando el mapeo global
        move = self.move_mapper.action_to_move(action, new_state)
        
        if move is None or move not in new_state.legal_moves:
            raise ValueError(f"Acción {action} no es legal en esta posición")
        
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
        """
        Retorna un vector binario indicando movimientos válidos.
        
        CAMBIO CRÍTICO: Ahora usa el mapeo global para marcar acciones válidas.
        
        Args:
            state: chess.Board actual
            
        Returns:
            Array numpy binario de forma (4672,) donde 1 = movimiento válido
        """
        return self.move_mapper.get_action_mask(state)
    
    def get_legal_actions(self, state):
        """
        Retorna lista de índices de acciones legales.
        
        Args:
            state: chess.Board
            
        Returns:
            List[int]: Índices de acciones (0-4671) que son legales
        """
        return self.move_mapper.get_legal_actions(state)
    
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
        """
        Cambia la perspectiva del tablero al jugador dado.
        En ajedrez con python-chess, el estado ya maneja esto internamente.
        """
        return state
    
    def get_encoded_state(self, state):
        """
        Codifica el estado del tablero como un array numpy.
        
        Formato: 12 canales de 8x8
        - Canales 0-5: Piezas blancas (Peón, Caballo, Alfil, Torre, Dama, Rey)
        - Canales 6-11: Piezas negras (Peón, Caballo, Alfil, Torre, Dama, Rey)
        
        Returns:
            Array numpy de forma (12, 8, 8) - formato PyTorch (canales primero)
        """
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
        """Imprime el tablero de forma legible"""
        print(state)
        print()
    
    def get_move_from_action(self, state, action):
        """
        Convierte un índice de acción en un movimiento de ajedrez.
        
        Args:
            state: chess.Board actual
            action: índice de acción (0-4671)
            
        Returns:
            chess.Move object
        """
        move = self.move_mapper.action_to_move(action, state)
        
        if move is None:
            raise ValueError(f"Acción {action} no corresponde a un movimiento legal")
        
        return move
    
    def get_action_from_move(self, move):
        """
        Convierte un movimiento chess.Move a un índice de acción.
        
        Args:
            move: chess.Move object
            
        Returns:
            int: índice de acción (0-4671)
        """
        return self.move_mapper.move_to_action(move)


# === FUNCIONES DE PRUEBA ===

def test_chess_game():
    """Prueba el sistema ChessGame con mapeo global."""
    print("=== PRUEBA DE CHESSGAME CON MAPEO GLOBAL ===\n")
    
    game = ChessGame()
    state = game.get_initial_state()
    
    print("1. Estado inicial:")
    game.render(state)
    
    # Verificar movimientos válidos
    valid_moves = game.get_valid_moves(state)
    legal_actions = game.get_legal_actions(state)
    
    print(f"Movimientos legales: {len(legal_actions)}")
    print(f"Máscara válida: {int(valid_moves.sum())} acciones marcadas")
    print(f"✓ Coinciden: {len(legal_actions) == int(valid_moves.sum())}")
    
    # Probar conversión bidireccional
    print("\n2. Conversión bidireccional:")
    move_e2e4 = chess.Move.from_uci("e2e4")
    action = game.get_action_from_move(move_e2e4)
    move_back = game.get_move_from_action(state, action)
    
    print(f"Movimiento → Acción → Movimiento")
    print(f"{move_e2e4.uci()} → {action} → {move_back.uci()}")
    print(f"✓ Consistente: {move_e2e4 == move_back}")
    
    # Aplicar movimiento
    print("\n3. Aplicar movimiento:")
    new_state = game.get_next_state(state, action, 1)
    game.render(new_state)
    
    # Verificar codificación
    print("4. Codificación de estado:")
    encoded = game.get_encoded_state(state)
    print(f"Shape: {encoded.shape}")
    print(f"Rango: [{encoded.min()}, {encoded.max()}]")
    print(f"Piezas totales: {int(encoded.sum())}")  # Debería ser 32
    
    print("\n✓ Todas las pruebas completadas")


def compare_old_vs_new():
    """
    Demuestra la diferencia entre el mapeo antiguo (local) y el nuevo (global).
    """
    print("\n=== COMPARACIÓN: MAPEO LOCAL vs GLOBAL ===\n")
    
    game = ChessGame()
    
    # Posición 1: Inicio
    state1 = chess.Board()
    actions1 = game.get_legal_actions(state1)
    
    print("Posición 1 (inicio):")
    print(f"Movimientos legales: {len(list(state1.legal_moves))}")
    print(f"Primeros 3 índices de acción: {actions1[:3]}")
    
    # Posición 2: Después de e2e4
    state2 = chess.Board()
    state2.push_uci("e2e4")
    actions2 = game.get_legal_actions(state2)
    
    print("\nPosición 2 (después de e2e4):")
    print(f"Movimientos legales: {len(list(state2.legal_moves))}")
    print(f"Primeros 3 índices de acción: {actions2[:3]}")
    
    print("\n📌 OBSERVACIÓN CLAVE:")
    print("Los índices de acción ahora son GLOBALES y NO dependen del orden")
    print("local de movimientos. Por ejemplo, 'e7e5' tendrá el MISMO índice")
    print("en cualquier posición donde sea legal.")
    
    # Verificar consistencia
    move_e7e5 = chess.Move.from_uci("e7e5")
    if move_e7e5 in state2.legal_moves:
        action = game.get_action_from_move(move_e7e5)
        print(f"\ne7e5 siempre será acción #{action} cuando sea legal")


if __name__ == "__main__":
    test_chess_game()
    compare_old_vs_new()