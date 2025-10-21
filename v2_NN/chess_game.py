import numpy as np
import chess
from move_mapping import MoveMapper

class ChessGame:
    """
    Interfaz del juego de ajedrez con mapeo global de movimientos.
    
    CONVENCIONES DE VALORES:
    - +1: Blancas ganan
    - -1: Negras ganan
    -  0: Empate
    
    Los valores siempre están desde la perspectiva del jugador ACTUAL (state.turn).
    """
    
    def __init__(self, include_queen_promotions=False):
        """
        Args:
            include_queen_promotions: Si True, usa mapeo extendido (4864 acciones)
                                     Si False, usa mapeo estándar (4672 acciones)
        """
        self.row_count = 8
        self.column_count = 8
        
        # Crear mapper
        self.move_mapper = MoveMapper(include_queen_promotions=include_queen_promotions)
        self.action_size = self.move_mapper.action_size
        
        print(f"✓ ChessGame inicializado")
        print(f"  Action size: {self.action_size}")
        print(f"  Promociones explícitas a dama: {include_queen_promotions}")
        
    def get_initial_state(self):
        """Retorna el estado inicial del tablero."""
        return chess.Board()
    
    def get_next_state(self, state, action, player):
        """
        Aplica una acción al estado y retorna el nuevo estado.
        
        NOTA: El parámetro 'player' existe por compatibilidad con la API genérica,
        pero no se usa en ajedrez (python-chess maneja turnos automáticamente).
        
        Args:
            state: chess.Board actual
            action: índice de acción (0 a action_size-1)
            player: 1 o -1 (ignorado en ajedrez)
            
        Returns:
            Nuevo chess.Board después de aplicar la acción
            
        Raises:
            ValueError: Si la acción no es legal
        """
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
        
        # Aplicar movimiento (state.turn cambia automáticamente)
        new_state.push(move)
        return new_state
    
    def get_valid_moves(self, state):
        """
        Retorna máscara binaria de movimientos válidos.
        
        Args:
            state: chess.Board actual
            
        Returns:
            numpy array de forma (action_size,) con 1.0 para acciones legales
        """
        return self.move_mapper.get_action_mask(state)
    
    def get_legal_actions(self, state):
        """
        Retorna lista de índices de acciones legales.
        
        Args:
            state: chess.Board
            
        Returns:
            List[int]: Índices de acciones legales
        """
        return self.move_mapper.get_legal_actions(state)
    
    def get_value_and_terminated(self, state, action_taken=None):
        """
        Retorna el valor del estado y si es terminal.
        
        CONVENCIÓN DE VALORES:
        - Si el juego terminó, el valor está desde la perspectiva del 
          jugador ACTUAL (state.turn), NO del que movió último.
        - Si es checkmate: value = -1 (el jugador actual perdió)
        - Si es empate: value = 0
        
        Args:
            state: chess.Board
            action_taken: No usado (por compatibilidad con API genérica)
            
        Returns:
            (value, is_terminal): tupla con valor en [-1, 0, 1] y booleano
        """
        # Checkmate
        if state.is_checkmate():
            # El jugador actual (state.turn) está en jaque mate → perdió
            return -1, True
        
        # Empates
        if state.is_stalemate():
            return 0, True
        if state.is_insufficient_material():
            return 0, True
        if state.is_seventyfive_moves():
            return 0, True
        if state.is_fivefold_repetition():
            return 0, True
        
        # Juego continúa
        return 0, False
    
    # ===================================================================
    # MÉTODOS ELIMINADOS (eran innecesarios para ajedrez con python-chess)
    # ===================================================================
    # - get_opponent(player): python-chess maneja turnos automáticamente
    # - get_opponent_value(value): causaba confusión con doble inversión
    # - change_perspective(state, player): no hace nada en ajedrez
    # ===================================================================
    
    def get_encoded_state(self, state):
        """
        Codifica el estado del tablero como un array numpy.
        
        Formato: 12 canales de 8x8 (piece-centric representation)
        - Canales 0-5: Piezas blancas (Peón, Caballo, Alfil, Torre, Dama, Rey)
        - Canales 6-11: Piezas negras (Peón, Caballo, Alfil, Torre, Dama, Rey)
        
        NOTA: La codificación es absoluta (no depende del turno).
        La fila 0 del tensor corresponde a la fila 8 del tablero (a8-h8).
        
        Returns:
            Array numpy de forma (12, 8, 8) - formato PyTorch (C, H, W)
        """
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
                # chess.SQUARES: 0=a1, 7=h1, 56=a8, 63=h8
                # Queremos: fila 0 del tensor = fila 8 del tablero
                row = 7 - (square // 8)
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
        """Imprime el tablero de forma legible."""
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


# === TESTS ===

def test_value_conventions():
    """
    Test para verificar que las convenciones de valores son correctas.
    """
    print("=" * 70)
    print("TEST: CONVENCIONES DE VALORES")
    print("=" * 70)
    
    game = ChessGame(include_queen_promotions=False)
    
    # Test 1: Mate para blancas
    print("\n1️⃣ Test: Mate para blancas")
    state_white_wins = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # Negras en mate
    game.render(state_white_wins)
    
    value, is_terminal = game.get_value_and_terminated(state_white_wins)
    print(f"  Turno: {'Blancas' if state_white_wins.turn else 'Negras'}")
    print(f"  Valor: {value}")
    print(f"  Terminal: {is_terminal}")
    print(f"  ✓ Correcto: value=-1 (jugador actual perdió)")
    
    assert value == -1 and is_terminal, "Valor incorrecto para mate"
    
    # Test 2: Mate para negras
    print("\n2️⃣ Test: Mate para negras")
    state_black_wins = chess.Board("r6k/6pp/8/8/8/8/8/K7 w - - 0 1")
    state_black_wins.push_uci("a1b1")  # Blancas mueven
    state_black_wins.push_uci("a8a1")  # Negras hacen mate
    game.render(state_black_wins)
    
    value, is_terminal = game.get_value_and_terminated(state_black_wins)
    print(f"  Turno: {'Blancas' if state_black_wins.turn else 'Negras'}")
    print(f"  Valor: {value}")
    print(f"  Terminal: {is_terminal}")
    print(f"  ✓ Correcto: value=-1 (jugador actual perdió)")
    
    assert value == -1 and is_terminal, "Valor incorrecto para mate"
    
    # Test 3: Empate
    print("\n3️⃣ Test: Empate (ahogado)")
    state_draw = chess.Board("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")
    game.render(state_draw)
    
    value, is_terminal = game.get_value_and_terminated(state_draw)
    print(f"  Turno: {'Blancas' if state_draw.turn else 'Negras'}")
    print(f"  Valor: {value}")
    print(f"  Terminal: {is_terminal}")
    print(f"  ✓ Correcto: value=0 (empate)")
    
    assert value == 0 and is_terminal, "Valor incorrecto para empate"
    
    # Test 4: Juego en progreso
    print("\n4️⃣ Test: Juego en progreso")
    state_ongoing = game.get_initial_state()
    game.render(state_ongoing)
    
    value, is_terminal = game.get_value_and_terminated(state_ongoing)
    print(f"  Turno: {'Blancas' if state_ongoing.turn else 'Negras'}")
    print(f"  Valor: {value}")
    print(f"  Terminal: {is_terminal}")
    print(f"  ✓ Correcto: value=0, terminal=False")
    
    assert value == 0 and not is_terminal, "Valor incorrecto para juego en progreso"
    
    print("\n" + "="*70)
    print("✅ TODAS LAS CONVENCIONES SON CORRECTAS")
    print("="*70)


def test_no_unnecessary_methods():
    """
    Verifica que los métodos innecesarios fueron eliminados.
    """
    print("\n" + "="*70)
    print("TEST: VERIFICAR LIMPIEZA DE MÉTODOS")
    print("="*70)
    
    game = ChessGame(include_queen_promotions=False)
    
    # Estos métodos NO deberían existir
    removed_methods = ['get_opponent', 'get_opponent_value', 'change_perspective']
    
    print("\nMétodos que deberían haber sido eliminados:")
    for method in removed_methods:
        has_method = hasattr(game, method)
        status = "❌ AÚN EXISTE" if has_method else "✅ ELIMINADO"
        print(f"  {method}: {status}")
    
    # Estos métodos SÍ deberían existir
    required_methods = [
        'get_initial_state',
        'get_next_state',
        'get_valid_moves',
        'get_legal_actions',
        'get_value_and_terminated',
        'get_encoded_state',
        'render',
        'get_move_from_action',
        'get_action_from_move'
    ]
    
    print("\nMétodos esenciales (deben existir):")
    all_present = True
    for method in required_methods:
        has_method = hasattr(game, method)
        status = "✅ PRESENTE" if has_method else "❌ FALTA"
        print(f"  {method}: {status}")
        if not has_method:
            all_present = False
    
    print("\n" + "="*70)
    if all_present:
        print("✅ INTERFAZ LIMPIA Y COMPLETA")
    else:
        print("❌ FALTAN MÉTODOS ESENCIALES")
    print("="*70)


def test_state_immutability():
    """
    Verifica que get_next_state no modifica el estado original.
    """
    print("\n" + "="*70)
    print("TEST: INMUTABILIDAD DE ESTADOS")
    print("="*70)
    
    game = ChessGame(include_queen_promotions=False)
    
    # Estado inicial
    state = game.get_initial_state()
    fen_original = state.fen()
    
    print(f"\nFEN original: {fen_original}")
    
    # Aplicar movimiento
    move = chess.Move.from_uci("e2e4")
    action = game.get_action_from_move(move)
    new_state = game.get_next_state(state, action, 1)
    
    fen_after = state.fen()
    fen_new = new_state.fen()
    
    print(f"FEN después de get_next_state: {fen_after}")
    print(f"FEN del nuevo estado: {fen_new}")
    
    if fen_original == fen_after:
        print("\n✅ Estado original NO fue modificado (correcto)")
    else:
        print("\n❌ Estado original FUE modificado (incorrecto)")
    
    if fen_original != fen_new:
        print("✅ Nuevo estado es diferente (correcto)")
    else:
        print("❌ Nuevo estado es igual al original (incorrecto)")
    
    print("\n" + "="*70)
    print("✅ TEST DE INMUTABILIDAD COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    test_value_conventions()
    test_no_unnecessary_methods()
    test_state_immutability()