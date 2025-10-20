import chess
import numpy as np


class MoveMapper:
    """
    Sistema de mapeo global entre movimientos de ajedrez y índices de acción.

    Inspirado en AlphaZero:
    - 64 casillas de origen × 73 posibles "planes de movimiento"
    - Total = 4672 acciones fijas.

    Cada casilla tiene:
    - 56 movimientos tipo reina  (8 direcciones × 7 distancias)
    - 8 movimientos de caballo    (saltos en L)
    - 9 movimientos de promoción  (3 direcciones × 3 piezas underpromo)
    
    CAMBIO CRÍTICO: Las promociones a dama se manejan como movimientos
    normales sin promoción explícita (compatibilidad con AlphaZero original).
    """

    def __init__(self, include_queen_promotions=False):
        """
        Args:
            include_queen_promotions: Si True, incluye promociones a dama explícitas
                                     aumentando action_size a 4864 (64×76)
        """
        self.include_queen_promotions = include_queen_promotions
        
        if include_queen_promotions:
            self.action_size = 4864  # 64 × 76 planes
            print("⚠️ Usando mapeo EXTENDIDO con promociones a dama explícitas")
        else:
            self.action_size = 4672  # 64 × 73 planes (AlphaZero estándar)
            print("✓ Usando mapeo ESTÁNDAR (promociones a dama implícitas)")
            
        self._move_to_index = {}
        self._index_to_move = {}
        self._build_move_mappings()

    def _build_move_mappings(self):
        action_idx = 0

        # Movimientos tipo reina (8 direcciones × 7 distancias = 56)
        queen_directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]

        # Movimientos de caballo (8 saltos)
        knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]

        # Piezas de promoción
        if self.include_queen_promotions:
            promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        else:
            # Solo underpromotions (Knight, Bishop, Rook)
            promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        for from_square in range(64):
            from_rank = from_square // 8
            from_file = from_square % 8

            # 1️⃣ Movimientos tipo reina (56 planes)
            for direction in queen_directions:
                for distance in range(1, 8):
                    to_file = from_file + direction[0] * distance
                    to_rank = from_rank + direction[1] * distance

                    if 0 <= to_file < 8 and 0 <= to_rank < 8:
                        to_square = to_rank * 8 + to_file
                        move_key = (from_square, to_square, None)
                        self._move_to_index[move_key] = action_idx
                        self._index_to_move[action_idx] = move_key
                    else:
                        self._index_to_move[action_idx] = None

                    action_idx += 1

            # 2️⃣ Movimientos de caballo (8 planes)
            for knight_move in knight_moves:
                to_file = from_file + knight_move[0]
                to_rank = from_rank + knight_move[1]

                if 0 <= to_file < 8 and 0 <= to_rank < 8:
                    to_square = to_rank * 8 + to_file
                    move_key = (from_square, to_square, None)
                    self._move_to_index[move_key] = action_idx
                    self._index_to_move[action_idx] = move_key
                else:
                    self._index_to_move[action_idx] = None

                action_idx += 1

            # 3️⃣ Movimientos de promoción
            num_promo_planes = len(promotion_pieces) * 3  # 9 o 12 planes
            
            if from_rank == 6:  # peón blanco en fila 7 (rank 6)
                directions = [(-1, 1), (0, 1), (1, 1)]  # Izq-adelante, adelante, Der-adelante
                promo_rank = 7
                
                for direction in directions:
                    for promo_piece in promotion_pieces:
                        to_file = from_file + direction[0]
                        to_rank = from_rank + direction[1]
                        
                        if 0 <= to_file < 8 and to_rank == promo_rank:
                            to_square = to_rank * 8 + to_file
                            move_key = (from_square, to_square, promo_piece)
                            self._move_to_index[move_key] = action_idx
                            self._index_to_move[action_idx] = move_key
                        else:
                            self._index_to_move[action_idx] = None
                        action_idx += 1

            elif from_rank == 1:  # peón negro en fila 2 (rank 1)
                directions = [(-1, -1), (0, -1), (1, -1)]  # Izq-atrás, atrás, Der-atrás
                promo_rank = 0
                
                for direction in directions:
                    for promo_piece in promotion_pieces:
                        to_file = from_file + direction[0]
                        to_rank = from_rank + direction[1]
                        
                        if 0 <= to_file < 8 and to_rank == promo_rank:
                            to_square = to_rank * 8 + to_file
                            move_key = (from_square, to_square, promo_piece)
                            self._move_to_index[move_key] = action_idx
                            self._index_to_move[action_idx] = move_key
                        else:
                            self._index_to_move[action_idx] = None
                        action_idx += 1
            else:
                # Rellenar planes de promoción para mantener 73 (o 76) movimientos por casilla
                for _ in range(num_promo_planes):
                    self._index_to_move[action_idx] = None
                    action_idx += 1

        expected_size = self.action_size
        assert action_idx == expected_size, \
            f"Se generaron {action_idx} acciones, se esperaban {expected_size}"
        
        print(f"✓ Mapeo de movimientos construido: {action_idx} acciones totales")
        print(f"✓ Movimientos únicos mapeados: {len(self._move_to_index)}")

    # === CONVERSIONES ===

    def move_to_action(self, move):
        """
        Convierte un chess.Move a un índice de acción.
        
        COMPORTAMIENTO MEJORADO:
        - Si include_queen_promotions=True: Mapea todas las promociones explícitamente
        - Si include_queen_promotions=False: Promociones a dama se tratan como None
        
        Args:
            move: chess.Move object
            
        Returns:
            int: índice de acción
            
        Raises:
            ValueError: Si el movimiento no está en el mapeo
        """
        promotion = move.promotion
        
        # Si no incluimos queens explícitas, tratarlas como None
        if not self.include_queen_promotions and promotion == chess.QUEEN:
            promotion = None

        move_key = (move.from_square, move.to_square, promotion)
        
        if move_key not in self._move_to_index:
            # Mensaje de error mejorado
            promo_str = chess.piece_name(promotion) if promotion else "None"
            raise ValueError(
                f"❌ Movimiento {move.uci()} no encontrado en mapeo.\n"
                f"   from_square={move.from_square} ({chess.square_name(move.from_square)})\n"
                f"   to_square={move.to_square} ({chess.square_name(move.to_square)})\n"
                f"   promotion={promo_str}\n"
                f"   include_queen_promotions={self.include_queen_promotions}"
            )
        
        return self._move_to_index[move_key]

    def action_to_move(self, action, state):
        """
        Convierte un índice de acción a un chess.Move.
        
        COMPORTAMIENTO MEJORADO:
        - Si el movimiento es de peón a última fila sin promoción explícita,
          asume promoción a dama (estándar de ajedrez)
        - Valida que el movimiento sea legal en el estado dado
        
        Args:
            action: índice de acción (0 a action_size-1)
            state: chess.Board actual
            
        Returns:
            chess.Move si es legal, None si no es legal o acción inválida
        """
        if action < 0 or action >= self.action_size:
            return None
            
        move_info = self._index_to_move.get(action)
        if move_info is None:
            return None

        from_sq, to_sq, promotion = move_info
        
        # Crear movimiento base
        move = chess.Move(from_sq, to_sq, promotion=promotion)

        # LÓGICA MEJORADA: Si es promoción de peón sin pieza explícita, asumir dama
        if promotion is None:
            piece = state.piece_at(from_sq)
            to_rank = to_sq // 8
            
            if piece and piece.piece_type == chess.PAWN:
                # Peón blanco llegando a fila 8 o peón negro llegando a fila 1
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    # En mapeo estándar, esto es promoción a dama implícita
                    move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

        # Validar legalidad
        if move in state.legal_moves:
            return move
        
        return None

    # === UTILIDADES ===

    def get_action_mask(self, state):
        """
        Retorna máscara binaria de acciones legales.
        
        Args:
            state: chess.Board
            
        Returns:
            numpy array de forma (action_size,) con 1.0 para acciones legales
        """
        mask = np.zeros(self.action_size, dtype=np.float32)
        
        for move in state.legal_moves:
            try:
                action = self.move_to_action(move)
                mask[action] = 1.0
            except ValueError as e:
                # Logging mejorado
                import sys
                print(f"⚠️ Advertencia: {e}", file=sys.stderr)
                continue
        
        return mask

    def get_legal_actions(self, state):
        """
        Retorna lista de índices de acciones legales.
        
        Args:
            state: chess.Board
            
        Returns:
            List[int]: índices de acciones legales
        """
        actions = []
        unmapped_moves = []
        
        for move in state.legal_moves:
            try:
                actions.append(self.move_to_action(move))
            except ValueError:
                unmapped_moves.append(move.uci())
                continue
        
        # Advertir si hay movimientos no mapeados (no debería pasar)
        if unmapped_moves:
            import sys
            print(f"⚠️ {len(unmapped_moves)} movimientos legales no mapeados: {unmapped_moves[:5]}", 
                  file=sys.stderr)
        
        return actions

    def get_statistics(self):
        """
        Retorna estadísticas del mapeo para debugging.
        
        Returns:
            dict con estadísticas
        """
        total_actions = self.action_size
        valid_actions = sum(1 for v in self._index_to_move.values() if v is not None)
        unique_moves = len(self._move_to_index)
        
        return {
            'action_size': total_actions,
            'valid_actions': valid_actions,
            'invalid_actions': total_actions - valid_actions,
            'unique_moves': unique_moves,
            'include_queen_promotions': self.include_queen_promotions,
            'density': unique_moves / total_actions
        }


# === TEST MEJORADO ===

def test_move_mapper_complete():
    """Prueba exhaustiva del sistema de mapeo con promociones."""
    print("=" * 70)
    print("PRUEBA EXHAUSTIVA DEL SISTEMA DE MAPEO")
    print("=" * 70)
    
    # Test con ambos modos
    for include_queen in [False, True]:
        print(f"\n{'='*70}")
        print(f"MODO: include_queen_promotions={include_queen}")
        print('='*70)
        
        mapper = MoveMapper(include_queen_promotions=include_queen)
        stats = mapper.get_statistics()
        
        print(f"\n📊 Estadísticas del mapeo:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test 1: Posición inicial
        print("\n1️⃣ Test: Posición inicial")
        board = chess.Board()
        mask = mapper.get_action_mask(board)
        legal_actions = mapper.get_legal_actions(board)
        
        print(f"  Movimientos legales (python-chess): {len(list(board.legal_moves))}")
        print(f"  Acciones válidas (máscara): {int(mask.sum())}")
        print(f"  Acciones legales (lista): {len(legal_actions)}")
        assert len(legal_actions) == int(mask.sum()), "❌ Inconsistencia máscara/lista"
        print("  ✓ Consistencia verificada")
        
        # Test 2: Movimiento simple
        print("\n2️⃣ Test: Conversión bidireccional (e2e4)")
        mv = chess.Move.from_uci("e2e4")
        act = mapper.move_to_action(mv)
        mv2 = mapper.action_to_move(act, board)
        print(f"  {mv.uci()} → acción {act} → {mv2.uci()}") # type: ignore
        assert mv == mv2, "❌ No coinciden"
        print("  ✓ Conversión correcta")
        
        # Test 3: Promociones blancas
        print("\n3️⃣ Test: Promociones blancas")
        board_w = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        print(f"  Posición: {board_w.fen()}")
        
        for move in sorted(board_w.legal_moves, key=lambda m: m.uci()):
            try:
                action = mapper.move_to_action(move)
                move_back = mapper.action_to_move(action, board_w)
                status = "✓" if move == move_back else "✗"
                promo_name = chess.piece_name(move.promotion) if move.promotion else "None"
                print(f"  {status} {move.uci():6s} (promo={promo_name:6s}) → acción {action:4d}")
            except ValueError as e:
                print(f"  ✗ {move.uci()} - ERROR: {e}")
        
        # Test 4: Promociones negras
        print("\n4️⃣ Test: Promociones negras")
        board_b = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
        print(f"  Posición: {board_b.fen()}")
        
        for move in sorted(board_b.legal_moves, key=lambda m: m.uci()):
            try:
                action = mapper.move_to_action(move)
                move_back = mapper.action_to_move(action, board_b)
                status = "✓" if move == move_back else "✗"
                promo_name = chess.piece_name(move.promotion) if move.promotion else "None"
                print(f"  {status} {move.uci():6s} (promo={promo_name:6s}) → acción {action:4d}")
            except ValueError as e:
                print(f"  ✗ {move.uci()} - ERROR: {e}")
        
        # Test 5: Todas las promociones posibles
        print("\n5️⃣ Test: Cobertura de promociones")
        promotion_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        test_board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        
        coverage = {}
        for promo in promotion_types:
            move = chess.Move(chess.A7, chess.A8, promotion=promo)
            try:
                action = mapper.move_to_action(move)
                coverage[chess.piece_name(promo)] = "✓ Mapeado"
            except ValueError:
                coverage[chess.piece_name(promo)] = "✗ No mapeado"
        
        for piece, status in coverage.items():
            print(f"  Promoción a {piece:6s}: {status}")
    
    print("\n" + "="*70)
    print("✅ PRUEBAS COMPLETADAS")
    print("="*70)


if __name__ == "__main__":
    test_move_mapper_complete()