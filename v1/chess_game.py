import numpy as np
import chess
from itertools import product

class ChessGame:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 4672
        
        # ⭐ CREAR MAPEO GLOBAL DE MOVIMIENTOS
        self.move_to_index, self.index_to_move = self._create_move_mapping()
    
    def _create_move_mapping(self):
        """
        Crea un mapeo GLOBAL entre movimientos y índices.
        
        Idea: Todos los movimientos posibles en ajedrez están mapeados
        a índices fijos (0-4671).
        
        Un movimiento en ajedrez es: origen (0-63) + destino (0-63)
        Pero no todos son legales.
        """
        move_to_index = {}
        index_to_move = {}
        
        idx = 0
        
        # Iterar sobre TODOS los cuadrados posibles
        for from_square in chess.SQUARES:  # 0-63
            for to_square in chess.SQUARES:  # 0-63
                if from_square == to_square:
                    continue  # No puede mover a la misma casilla
                
                # Crear movimiento
                move = chess.Move(from_square, to_square)
                move_uci = move.uci()
                
                # Mapear
                move_to_index[move_uci] = idx
                index_to_move[idx] = move_uci
                
                idx += 1
        
        print(f"✓ Mapeo global creado: {len(move_to_index)} movimientos únicos")
        
        return move_to_index, index_to_move
    
    def get_valid_moves(self, state):
        """
        Retorna un vector binario usando el MAPEO GLOBAL.
        """
        valid_moves = np.zeros(self.action_size)
        legal_moves = list(state.legal_moves)
        
        # ⭐ USAR MAPEO GLOBAL
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci in self.move_to_index:
                idx = self.move_to_index[move_uci]
                valid_moves[idx] = 1
        
        return valid_moves
    
    def get_move_from_action(self, state, action):
        """
        Convertir índice a movimiento usando el MAPEO GLOBAL.
        """
        # Obtener el movimiento UCI del mapeo
        move_uci = self.index_to_move[action]
        
        # Convertir a objeto chess.Move
        move = chess.Move.from_uci(move_uci)
        
        # Verificar que es legal (seguridad)
        if move not in state.legal_moves:
            raise ValueError(f"Movimiento {move_uci} no es legal en esta posición")
        
        return move
    
    def get_next_state(self, state, action, player):
        """
        Aplicar movimiento. Ahora action es GLOBALMENTE consistente.
        """
        move = self.get_move_from_action(state, action)
        new_state = state.copy()
        new_state.push(move)
        return new_state
    
    # ... resto de métodos ...