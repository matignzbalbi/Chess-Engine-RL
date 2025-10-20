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
    - 9 movimientos de promoción  (solo para peones en fila de promoción y solo knight/bishop/rook)

    Las promociones a dama se manejan implícitamente en action_to_move.
    """

    def __init__(self):
        self.action_size = 4672
        self._move_to_index = {}
        self._index_to_move = {}
        self._build_move_mappings()

    def _build_move_mappings(self):
        action_idx = 0

        # Movimientos tipo reina (8 direcciones)
        queen_directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]

        # Movimientos de caballo (8 saltos)
        knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]

        # Piezas de underpromotion (sin dama)
        underpromotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        for from_square in range(64):
            from_rank = from_square // 8
            from_file = from_square % 8

            # 1️⃣ Movimientos tipo reina
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

            # 2️⃣ Movimientos de caballo
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

            # 3️⃣ Movimientos de promoción (solo knight/bishop/rook)
            if from_rank == 6:  # peón blanco
                directions = [(-1, 1), (0, 1), (1, 1)]
                promo_rank = 7
                for direction in directions:
                    for promo_piece in underpromotion_pieces:
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

            elif from_rank == 1:  # peón negro
                directions = [(-1, -1), (0, -1), (1, -1)]
                promo_rank = 0
                for direction in directions:
                    for promo_piece in underpromotion_pieces:
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
                # Rellenar 9 posiciones para mantener 73 movimientos por casilla
                for _ in range(9):
                    self._index_to_move[action_idx] = None
                    action_idx += 1

        assert action_idx == 4672, f"Se generaron {action_idx} acciones, esperaban 4672"
        print(f"✓ Mapeo de movimientos construido: {action_idx} acciones totales")
        print(f"✓ Movimientos únicos mapeados: {len(self._move_to_index)}")

    # === CONVERSIONES ===

    def move_to_action(self, move):
        # Las promociones a dama se manejan como None
        promotion = move.promotion
        if promotion == chess.QUEEN:
            promotion = None

        move_key = (move.from_square, move.to_square, promotion)
        if move_key not in self._move_to_index:
            raise ValueError(
                f"Movimiento {move.uci()} no encontrado en mapeo. "
                f"from={move.from_square}, to={move.to_square}, promo={move.promotion}"
            )
        return self._move_to_index[move_key]

    def action_to_move(self, action, state):
        move_info = self._index_to_move.get(action)
        if move_info is None:
            return None

        from_sq, to_sq, promotion = move_info
        move = chess.Move(from_sq, to_sq, promotion=promotion)

        # Si es peón que llega a última fila sin promoción explícita, asumir dama
        if move.promotion is None:
            piece = state.piece_at(from_sq)
            to_rank = to_sq // 8
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

        if move in state.legal_moves:
            return move
        return None

    # === UTILIDADES ===

    def get_action_mask(self, state):
        mask = np.zeros(self.action_size, dtype=np.float32)
        for move in state.legal_moves:
            try:
                action = self.move_to_action(move)
                mask[action] = 1.0
            except ValueError as e:
                import sys
                print(f"⚠️ Movimiento {move.uci()} no mapeado: {e}", file=sys.stderr)
        return mask

    def get_legal_actions(self, state):
        actions = []
        for move in state.legal_moves:
            try:
                actions.append(self.move_to_action(move))
            except ValueError:
                continue
        return actions


# === TEST AUTOMÁTICO ===

def test_move_mapper():
    print("=== PRUEBA DEL SISTEMA DE MAPEO ===\n")

    mapper = MoveMapper()
    print(f"Total acciones: {mapper.action_size}")
    print(f"Movimientos únicos mapeados: {len(mapper._move_to_index)}")

    board = chess.Board()
    mask = mapper.get_action_mask(board)
    print("\nPosición inicial:")
    print(f"  Movimientos legales: {len(list(board.legal_moves))}")
    print(f"  Acciones válidas: {int(mask.sum())}")

    # Test: movimiento simple
    mv = chess.Move.from_uci("e2e4")
    act = mapper.move_to_action(mv)
    mv2 = mapper.action_to_move(act, board)
    print(f"\nConversión e2e4 → acción {act} → {mv2}")
    print("  ✓ Coinciden" if mv == mv2 else "  ✗ No coinciden")

    # Test: promoción blanca
    board_w = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    for move in board_w.legal_moves:
        a = mapper.move_to_action(move)
        print(f"  Blanca: {move.uci()} → acción {a}")

    # Test: promoción negra
    board_b = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
    for move in board_b.legal_moves:
        a = mapper.move_to_action(move)
        print(f"  Negra: {move.uci()} → acción {a}")

    print("\n✓ Pruebas completadas.")


if __name__ == "__main__":
    test_move_mapper()