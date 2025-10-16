import chess
import numpy as np

class MoveMapper:
    """
    Sistema de mapeo global entre movimientos de ajedrez y índices de acción.
    
    Inspirado en AlphaZero, crea un mapeo fijo donde cada casilla de origen
    tiene 73 "planes de movimiento" posibles:
    - 56 movimientos tipo reina (8 direcciones × 7 distancias)
    - 8 movimientos de caballo
    - 9 subpromociones (3 direcciones × 3 tipos de pieza)
    
    Total: 64 casillas × 73 planes = 4672 acciones
    """
    
    def __init__(self):
        self.action_size = 4672
        
        # Mapeos bidireccionales
        self._move_to_index = {}
        self._index_to_move = {}
        
        self._build_move_mappings()
    
    def _build_move_mappings(self):
        """
        Construye el mapeo COMPLETO de 4672 acciones.
        
        Estructura: Para cada una de las 64 casillas, asignamos 73 "planes":
        - Planes 0-55: Movimientos tipo reina (8 direcciones × 7 distancias)
        - Planes 56-63: Movimientos de caballo (8 saltos)
        - Planes 64-72: Subpromociones (3 direcciones × 3 tipos)
        
        Fórmula: action_index = from_square * 73 + plan_index
        """
        action_idx = 0
        
        # === DEFINIR DIRECCIONES ===
        
        # Movimientos tipo reina: 8 direcciones
        queen_directions = [
            (0, 1),   # Norte
            (1, 1),   # Noreste
            (1, 0),   # Este
            (1, -1),  # Sureste
            (0, -1),  # Sur
            (-1, -1), # Suroeste
            (-1, 0),  # Oeste
            (-1, 1)   # Noroeste
        ]
        
        # Movimientos de caballo: 8 saltos en L
        knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]
        
        # Direcciones de promoción (desde perspectiva de blancas)
        # Para peones en fila 6 → 7: avanzar, capturar izq, capturar der
        underpromotion_directions = [
            (-1, 1),  # Captura izquierda
            (0, 1),   # Avance
            (1, 1)    # Captura derecha
        ]
        
        # Tipos de subpromoción (promoción a dama se maneja como movimiento reina)
        underpromotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        
        # === ITERAR SOBRE LAS 64 CASILLAS ===
        for from_square in range(64):
            from_rank = from_square // 8  # 0-7 (fila)
            from_file = from_square % 8   # 0-7 (columna)
            
            # Plan 0-55: Movimientos tipo reina (8 direcciones × 7 distancias)
            for direction in queen_directions:
                for distance in range(1, 8):
                    to_file = from_file + direction[0] * distance
                    to_rank = from_rank + direction[1] * distance
                    
                    # Guardar el movimiento si está dentro del tablero
                    if 0 <= to_file < 8 and 0 <= to_rank < 8:
                        to_square = to_rank * 8 + to_file
                        move_key = (from_square, to_square, None)
                        self._move_to_index[move_key] = action_idx
                        self._index_to_move[action_idx] = move_key
                    
                    # Incrementar SIEMPRE, incluso si está fuera del tablero
                    action_idx += 1
            
            # Plan 56-63: Movimientos de caballo (8 saltos)
            for knight_move in knight_moves:
                to_file = from_file + knight_move[0]
                to_rank = from_rank + knight_move[1]
                
                if 0 <= to_file < 8 and 0 <= to_rank < 8:
                    to_square = to_rank * 8 + to_file
                    move_key = (from_square, to_square, None)
                    self._move_to_index[move_key] = action_idx
                    self._index_to_move[action_idx] = move_key
                
                action_idx += 1
            
            # Plan 64-72: Subpromociones (9 combinaciones)
            # Mapeamos TODAS las promociones posibles desde cualquier fila
            # Esto cubre casos especiales y variantes de ajedrez
            for direction in underpromotion_directions:
                for promo_piece in underpromotion_pieces:
                    to_file = from_file + direction[0]
                    to_rank = from_rank + direction[1]
                    
                    # Mapear si el destino está dentro del tablero
                    if 0 <= to_file < 8 and 0 <= to_rank < 8:
                        to_square = to_rank * 8 + to_file
                        
                        # Mapear promoción de blancas a fila 7
                        if to_rank == 7:
                            move_key = (from_square, to_square, promo_piece)
                            self._move_to_index[move_key] = action_idx
                            self._index_to_move[action_idx] = move_key
                        
                        # Mapear promoción de negras a fila 0
                        elif to_rank == 0:
                            move_key = (from_square, to_square, promo_piece)
                            self._move_to_index[move_key] = action_idx
                            self._index_to_move[action_idx] = move_key
                    
                    action_idx += 1
        
        assert action_idx == 4672, f"Se generaron {action_idx} acciones, esperaban 4672"
        print(f"✓ Mapeo de movimientos construido: {action_idx} acciones totales")
        print(f"✓ Movimientos únicos mapeados: {len(self._move_to_index)}")
    
    def move_to_action(self, move):
        """
        Convierte un movimiento de chess.Move a un índice de acción.
        
        Args:
            move: chess.Move object
            
        Returns:
            int: Índice de acción (0-4671)
        """
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        
        # Para promociones a dama, usar None (se mapea como movimiento tipo reina)
        if promotion == chess.QUEEN:
            promotion = None
        
        move_key = (from_square, to_square, promotion)
        
        if move_key not in self._move_to_index:
            raise ValueError(
                f"Movimiento {move.uci()} no encontrado en mapeo. "
                f"from={from_square}, to={to_square}, promo={promotion}"
            )
        
        return self._move_to_index[move_key]
    
    def action_to_move(self, action, state):
        """
        Convierte un índice de acción a un movimiento chess.Move.
        
        Args:
            action: int, índice de acción (0-4671)
            state: chess.Board, tablero actual (para validar)
            
        Returns:
            chess.Move object o None si el movimiento es ilegal
        """
        if action not in self._index_to_move:
            return None
        
        from_square, to_square, promotion = self._index_to_move[action]
        
        # Crear movimiento
        move = chess.Move(from_square, to_square, promotion=promotion)
        
        # Verificar si es legal en la posición actual
        if move in state.legal_moves:
            return move
        
        return None
    
    def get_action_mask(self, state):
        """
        Crea una máscara binaria de acciones válidas para el estado dado.
        
        Args:
            state: chess.Board
            
        Returns:
            np.array de forma (4672,) con 1s en movimientos legales
        """
        mask = np.zeros(self.action_size, dtype=np.float32)
        
        for move in state.legal_moves:
            try:
                action = self.move_to_action(move)
                mask[action] = 1.0
            except ValueError as e:
                # Movimiento no mapeado - esto puede ocurrir con variantes especiales
                # Lo reportamos pero continuamos
                import sys
                print(f"⚠️  Movimiento {move.uci()} no mapeado: {e}", file=sys.stderr)
                continue
        
        return mask
    
    def get_legal_actions(self, state):
        """
        Retorna lista de índices de acciones legales.
        
        Args:
            state: chess.Board
            
        Returns:
            List[int]: Índices de acciones legales
        """
        legal_actions = []
        
        for move in state.legal_moves:
            try:
                action = self.move_to_action(move)
                legal_actions.append(action)
            except ValueError:
                continue
        
        return legal_actions


# === FUNCIONES DE UTILIDAD ===

def test_move_mapper():
    """Prueba el sistema de mapeo de movimientos."""
    print("=== PRUEBA DEL SISTEMA DE MAPEO ===\n")
    
    mapper = MoveMapper()
    
    print(f"Total de acciones en el mapeo: {mapper.action_size}")
    print(f"Movimientos únicos almacenados: {len(mapper._move_to_index)}\n")
    
    # Prueba 1: Posición inicial
    print("1. Posición inicial:")
    board = chess.Board()
    print(board)
    print()
    
    legal_actions = mapper.get_legal_actions(board)
    print(f"Movimientos legales: {len(list(board.legal_moves))}")
    print(f"Acciones mapeadas: {len(legal_actions)}")
    print(f"Primeros 5 índices: {legal_actions[:5]}")
    
    # Prueba 2: Verificar que todos los movimientos se mapean
    print("\n2. Verificación de cobertura:")
    unmapped_count = 0
    for move in board.legal_moves:
        try:
            action = mapper.move_to_action(move)
        except ValueError:
            unmapped_count += 1
            print(f"  ⚠️  No mapeado: {move.uci()}")
    
    if unmapped_count == 0:
        print("  ✓ Todos los movimientos están mapeados")
    else:
        print(f"  ✗ {unmapped_count} movimientos sin mapear")
    
    # Prueba 3: Conversión bidireccional
    print("\n3. Conversión movimiento → acción → movimiento:")
    move = chess.Move.from_uci("e2e4")
    action = mapper.move_to_action(move)
    move_back = mapper.action_to_move(action, board)
    print(f"  Original: {move.uci()}")
    print(f"  Índice: {action}")
    print(f"  Recuperado: {move_back.uci() if move_back else 'None'}")
    print(f"  ✓ Coinciden: {move == move_back}")
    
    # Prueba 4: Máscara de acciones
    print("\n4. Máscara de acciones válidas:")
    mask = mapper.get_action_mask(board)
    print(f"  Dimensión del vector: {len(mask)}")
    print(f"  Acciones válidas: {int(mask.sum())}")
    print(f"  Acciones inválidas: {int((1 - mask).sum())}")
    
    # Prueba 5: Promoción
    print("\n5. Prueba de promoción:")
    board_promo = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    print(board_promo)
    legal_promos = list(board_promo.legal_moves)
    print(f"  Movimientos disponibles: {len(legal_promos)}")
    
    for move in legal_promos:
        try:
            action = mapper.move_to_action(move)
            print(f"    {move.uci()} → acción {action}")
        except ValueError as e:
            print(f"    ⚠️  {move.uci()} → ERROR: {e}")
    
    print("\n✓ Pruebas completadas")


if __name__ == "__main__":
    test_move_mapper()