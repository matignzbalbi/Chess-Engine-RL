import numpy as np
import chess
from move_mapping import MoveMapper

class ChessGame:
    """
    Interfaz del juego de ajedrez con mapeo global de movimientos.
    
    ACTUALIZACIÓN: Ahora soporta dos modos de promoción:
    - Estándar (4672 acciones): Promociones a dama implícitas
    - Extendido (4864 acciones): Todas las promociones explícitas
    """
    
    def __init__(self, include_queen_promotions=False):
        """
        Args:
            include_queen_promotions: Si True, usa mapeo extendido con queens explícitas
        """
        self.row_count = 8
        self.column_count = 8
        
        # Crear mapper con configuración elegida
        self.move_mapper = MoveMapper(include_queen_promotions=include_queen_promotions)
        self.action_size = self.move_mapper.action_size
        
        print(f"✓ ChessGame inicializado")
        print(f"  Action size: {self.action_size}")
        print(f"  Promociones explícitas a dama: {include_queen_promotions}")
        
    def get_initial_state(self):
        """Retorna el estado inicial del tablero"""
        return chess.Board()
    
    def get_next_state(self, state, action, player):
        """
        Aplica una acción al estado y retorna el nuevo estado.
        
        Args:
            state: chess.Board actual
            action: índice de acción global (0 a action_size-1)
            player: 1 para blancas, -1 para negras (no usado por python-chess)
            
        Returns:
            Nuevo estado después de aplicar la acción
            
        Raises:
            ValueError: Si la acción no es legal
        """
        new_state = state.copy()
        
        # Convertir acción a movimiento usando el mapeo global
        move = self.move_mapper.action_to_move(action, new_state)
        
        if move is None:
            raise ValueError(
                f"Acción {action} no corresponde a un movimiento legal en esta posición.\n"
                f"FEN: {state.fen()}"
            )
        
        if move not in new_state.legal_moves:
            raise ValueError(
                f"Movimiento {move.uci()} (acción {action}) no es legal.\n"
                f"FEN: {state.fen()}"
            )
        
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
        """
        Retorna un vector binario indicando movimientos válidos.
        
        Args:
            state: chess.Board actual
            
        Returns:
            Array numpy binario de forma (action_size,) donde 1 = movimiento válido
        """
        return self.move_mapper.get_action_mask(state)
    
    def get_legal_actions(self, state):
        """
        Retorna lista de índices de acciones legales.
        
        Args:
            state: chess.Board
            
        Returns:
            List[int]: Índices de acciones que son legales
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
    
    def get_value_and_terminated(self, state, action_taken=None):
        """
        Retorna el valor del estado y si es terminal.
        
        Args:
            state: chess.Board
            action_taken: Último movimiento realizado (no usado, por compatibilidad)
            
        Returns:
            (value, is_terminal): tupla con valor (-1, 0, 1) y booleano
        """
        if state.is_checkmate():
            # El jugador actual (cuyo turno es) está en jaque mate
            value = -1  # Perdió el jugador actual
            return value, True
        
        # Empates
        if state.is_stalemate():
            return 0, True
        if state.is_insufficient_material():
            return 0, True
        if state.is_seventyfive_moves():
            return 0, True
        if state.is_fivefold_repetition():
            return 0, True
        if state.can_claim_threefold_repetition():
            # Opcional: considerar empate por repetición
            return 0, True
        
        # Juego continúa
        return 0, False
    
    def get_opponent(self, player):
        """Retorna el oponente del jugador (1 → -1, -1 → 1)"""
        return -player
    
    def get_opponent_value(self, value):
        """Invierte el valor desde la perspectiva del oponente"""
        return -value
    
    def change_perspective(self, state, player):
        """
        Cambia la perspectiva del tablero al jugador dado.
        
        En ajedrez con python-chess, el estado ya maneja esto internamente
        a través de state.turn, así que simplemente retornamos el estado.
        
        NOTA: Esta función existe por compatibilidad con la API genérica
        de juegos (Connect4, Tic-Tac-Toe), pero en ajedrez es un no-op.
        """
        return state
    
    def get_encoded_state(self, state):
        """
        Codifica el estado del tablero como un array numpy.
        
        Formato: 12 canales de 8x8
        - Canales 0-5: Piezas blancas (Peón, Caballo, Alfil, Torre, Dama, Rey)
        - Canales 6-11: Piezas negras (Peón, Caballo, Alfil, Torre, Dama, Rey)
        
        NOTA: La codificación es desde la perspectiva de las blancas.
        La fila 0 del tensor corresponde a la fila 8 del tablero (a8-h8).
        
        Returns:
            Array numpy de forma (12, 8, 8) - formato PyTorch (canales primero)
        """
        # Formato: (canales, filas, columnas) para PyTorch
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
                # chess.SQUARES va de 0 (a1) a 63 (h8)
                # Queremos que fila 0 del tensor = fila 8 del tablero
                row = 7 - (square // 8)  # Invertir fila
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
            action: índice de acción (0 a action_size-1)
            
        Returns:
            chess.Move object
            
        Raises:
            ValueError: Si la acción no es válida
        """
        move = self.move_mapper.action_to_move(action, state)
        
        if move is None:
            raise ValueError(
                f"Acción {action} no corresponde a un movimiento legal.\n"
                f"FEN: {state.fen()}"
            )
        
        return move
    
    def get_action_from_move(self, move):
        """
        Convierte un movimiento chess.Move a un índice de acción.
        
        Args:
            move: chess.Move object
            
        Returns:
            int: índice de acción (0 a action_size-1)
            
        Raises:
            ValueError: Si el movimiento no está en el mapeo
        """
        return self.move_mapper.move_to_action(move)


# === FUNCIONES DE PRUEBA ===

def test_promotions_comprehensive():
    """Prueba exhaustiva de promociones en ambos modos."""
    print("=" * 70)
    print("PRUEBA EXHAUSTIVA DE PROMOCIONES")
    print("=" * 70)
    
    for include_queen in [False, True]:
        print(f"\n{'='*70}")
        print(f"MODO: include_queen_promotions={include_queen}")
        print('='*70)
        
        game = ChessGame(include_queen_promotions=include_queen)
        
        # Test 1: Promoción de peón blanco
        print("\n1️⃣ Test: Promoción de peón blanco")
        state = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        game.render(state)
        
        valid_moves = game.get_valid_moves(state)
        legal_actions = game.get_legal_actions(state)
        
        print(f"  Movimientos legales: {len(list(state.legal_moves))}")
        print(f"  Acciones válidas: {int(valid_moves.sum())}")
        print(f"  Acciones legales: {len(legal_actions)}")
        
        print("\n  Detalle de movimientos:")
        for move in sorted(state.legal_moves, key=lambda m: m.uci()):
            try:
                action = game.get_action_from_move(move)
                promo = chess.piece_name(move.promotion) if move.promotion else "None"
                print(f"    ✓ {move.uci():6s} (promo={promo:6s}) → acción {action}")
            except ValueError as e:
                print(f"    ✗ {move.uci()} - ERROR: {e}")
        
        # Test 2: Aplicar promoción
        print("\n2️⃣ Test: Aplicar promoción a dama")
        move_a8q = chess.Move.from_uci("a7a8q")
        try:
            action = game.get_action_from_move(move_a8q)
            print(f"  Movimiento a7a8q → acción {action}")
            
            new_state = game.get_next_state(state, action, 1)
            print(f"  Estado después de promoción:")
            game.render(new_state)
            
            # Verificar que hay una dama en a8
            piece_at_a8 = new_state.piece_at(chess.A8)
            if piece_at_a8 and piece_at_a8.piece_type == chess.QUEEN:
                print("  ✓ Dama correctamente creada en a8")
            else:
                print("  ✗ ERROR: No hay dama en a8")
                
        except ValueError as e:
            print(f"  ✗ ERROR al procesar a7a8q: {e}")
        
        # Test 3: Underpromotion
        print("\n3️⃣ Test: Underpromotion (a7a8n)")
        move_a8n = chess.Move.from_uci("a7a8n")
        try:
            action = game.get_action_from_move(move_a8n)
            print(f"  Movimiento a7a8n → acción {action}")
            
            new_state = game.get_next_state(state, action, 1)
            piece_at_a8 = new_state.piece_at(chess.A8)
            if piece_at_a8 and piece_at_a8.piece_type == chess.KNIGHT:
                print("  ✓ Caballo correctamente creado en a8")
            else:
                print("  ✗ ERROR: No hay caballo en a8")
                
        except ValueError as e:
            print(f"  ✗ ERROR: {e}")
        
        # Test 4: Promoción de peón negro
        print("\n4️⃣ Test: Promoción de peón negro")
        state_black = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
        game.render(state_black)
        
        print(f"  Movimientos legales: {len(list(state_black.legal_moves))}")
        
        print("\n  Detalle de movimientos:")
        for move in sorted(state_black.legal_moves, key=lambda m: m.uci()):
            try:
                action = game.get_action_from_move(move)
                promo = chess.piece_name(move.promotion) if move.promotion else "None"
                print(f"    ✓ {move.uci():6s} (promo={promo:6s}) → acción {action}")
            except ValueError as e:
                print(f"    ✗ {move.uci()} - ERROR")
    
    print("\n" + "="*70)
    print("✅ PRUEBAS DE PROMOCIONES COMPLETADAS")
    print("="*70)


def test_full_game_with_promotion():
    """Simula una partida corta que termina en promoción."""
    print("\n" + "="*70)
    print("SIMULACIÓN: Partida con promoción")
    print("="*70)
    
    game = ChessGame(include_queen_promotions=False)
    state = game.get_initial_state()
    
    # Secuencia de movimientos que llevan a una promoción rápida
    moves_uci = ["e2e4", "d7d5", "e4d5", "e7e6", "d5e6", "f7e6", 
                 "d2d4", "g8f6", "d4d5", "e6d5", "d1d5"]
    
    print("\nJugando secuencia de movimientos...")
    for i, move_uci in enumerate(moves_uci, 1):
        move = chess.Move.from_uci(move_uci)
        action = game.get_action_from_move(move)
        
        player_name = "Blancas" if state.turn else "Negras"
        print(f"{i}. {player_name}: {move_uci} (acción {action})")
        
        state = game.get_next_state(state, action, 1)
    
    print("\nPosición alcanzada:")
    game.render(state)
    
    # Verificar movimientos válidos
    valid_moves = game.get_valid_moves(state)
    print(f"\nMovimientos válidos disponibles: {int(valid_moves.sum())}")
    
    print("\n✅ Simulación completada")


def compare_action_spaces():
    """Compara el espacio de acciones entre ambos modos."""
    print("\n" + "="*70)
    print("COMPARACIÓN DE ESPACIOS DE ACCIÓN")
    print("="*70)
    
    game_standard = ChessGame(include_queen_promotions=False)
    game_extended = ChessGame(include_queen_promotions=True)
    
    # Posición con múltiples promociones
    state = chess.Board("4k3/PPPPPPPP/8/8/8/8/pppppppp/4K3 w - - 0 1")
    
    print("\nPosición de prueba (múltiples peones en fila de promoción):")
    print(state)
    
    # Comparar espacios
    valid_standard = game_standard.get_valid_moves(state)
    valid_extended = game_extended.get_valid_moves(state)
    
    print(f"\nMODO ESTÁNDAR (4672 acciones):")
    print(f"  Acciones válidas: {int(valid_standard.sum())}")
    print(f"  Densidad: {valid_standard.sum() / game_standard.action_size:.2%}")
    
    print(f"\nMODO EXTENDIDO (4864 acciones):")
    print(f"  Acciones válidas: {int(valid_extended.sum())}")
    print(f"  Densidad: {valid_extended.sum() / game_extended.action_size:.2%}")
    
    diff = int(valid_extended.sum()) - int(valid_standard.sum())
    print(f"\nDiferencia: {diff} acciones adicionales en modo extendido")
    print("  (Estas son las promociones a dama explícitas)")
    
    print("\n📊 Recomendación:")
    print("  - Modo ESTÁNDAR: Más eficiente, compatible con AlphaZero original")
    print("  - Modo EXTENDIDO: Más explícito, mejor para debugging")


if __name__ == "__main__":
    test_promotions_comprehensive()
    test_full_game_with_promotion()
    compare_action_spaces()