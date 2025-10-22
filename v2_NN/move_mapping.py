import chess
import numpy as np


class MoveMapper:
  

    def __init__(self, include_queen_promotions=False):
       
        self.include_queen_promotions = include_queen_promotions
        
        if include_queen_promotions:
            self.action_size = 4864  # 64 × 76 planes
            print("Usando mapeo EXTENDIDO con promociones a dama explícitas")
        else:
            self.action_size = 4672  # 64 × 73 planes (AlphaZero estándar)
            print("Usando mapeo ESTÁNDAR (promociones a dama implícitas)")
            
        self._move_to_index = {}
        self._index_to_move = {}
        self._build_move_mappings()

    def _build_move_mappings(self):
        action_idx = 0

        queen_directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]

        knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]

        if self.include_queen_promotions:
            promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        else:
            # Solo underpromotions (Knight, Bishop, Rook)
            promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        for from_square in range(64):
            from_rank = from_square // 8
            from_file = from_square % 8

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

            num_promo_planes = len(promotion_pieces) * 3  
            
            if from_rank == 6:  
                directions = [(-1, 1), (0, 1), (1, 1)] 
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

            elif from_rank == 1:  
                directions = [(-1, -1), (0, -1), (1, -1)]  
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
                
                for _ in range(num_promo_planes):
                    self._index_to_move[action_idx] = None
                    action_idx += 1

        expected_size = self.action_size
        assert action_idx == expected_size, \
            f"Se generaron {action_idx} acciones, se esperaban {expected_size}"
        
        print(f"Mapeo de movimientos construido: {action_idx} acciones totales")
        print(f"Movimientos únicos mapeados: {len(self._move_to_index)}")


    def move_to_action(self, move):
        
        promotion = move.promotion
        
        # Si no incluimos queens explícitas, tratarlas como None
        if not self.include_queen_promotions and promotion == chess.QUEEN:
            promotion = None

        move_key = (move.from_square, move.to_square, promotion)
        
        if move_key not in self._move_to_index:
            # Mensaje de error mejorado
            promo_str = chess.piece_name(promotion) if promotion else "None"
            raise ValueError(
                f"Movimiento {move.uci()} no encontrado en mapeo.\n"
                f"from_square={move.from_square} ({chess.square_name(move.from_square)})\n"
                f"to_square={move.to_square} ({chess.square_name(move.to_square)})\n"
                f"promotion={promo_str}\n"
                f"include_queen_promotions={self.include_queen_promotions}"
            )
        
        return self._move_to_index[move_key]

    def action_to_move(self, action, state):
      
        if action < 0 or action >= self.action_size:
            return None
            
        move_info = self._index_to_move.get(action)
        if move_info is None:
            return None

        from_sq, to_sq, promotion = move_info
        
        # Crear movimiento base
        move = chess.Move(from_sq, to_sq, promotion=promotion)

        if promotion is None:
            piece = state.piece_at(from_sq)
            to_rank = to_sq // 8
            
            if piece and piece.piece_type == chess.PAWN:
              
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                  
                    move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

        if move in state.legal_moves:
            return move
        
        return None
    
    def get_action_mask(self, state):
    
        mask = np.zeros(self.action_size, dtype=np.float32)
        
        for move in state.legal_moves:
            try:
                action = self.move_to_action(move)
                mask[action] = 1.0
            except ValueError as e:
                # Logging mejorado
                import sys
                print(f"Advertencia: {e}", file=sys.stderr)
                continue
        
        return mask

    def get_legal_actions(self, state):
       
        actions = []
        unmapped_moves = []
        
        for move in state.legal_moves:
            try:
                actions.append(self.move_to_action(move))
            except ValueError:
                unmapped_moves.append(move.uci())
                continue
        
        # Advertir si hay movimientos no mapeados
        if unmapped_moves:
            import sys
            print(f"{len(unmapped_moves)} movimientos legales no mapeados: {unmapped_moves[:5]}", 
                  file=sys.stderr)
        
        return actions

    def get_statistics(self):
       
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

