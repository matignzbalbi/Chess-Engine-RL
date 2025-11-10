import csv
import os
import numpy as np
from datetime import datetime
import json
import chess


try:
    from database import ChessDatabase
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Módulo 'database' no encontrado.")


class GameLogger:
    
    def __init__(self, log_dir="game_logs", use_database=True):

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Archivos CSV (mantener como backup)
        self.moves_file = os.path.join(log_dir, "games_moves.csv")
        self.training_file = os.path.join(log_dir, "training_data.csv")
        self.stats_file = os.path.join(log_dir, "game_stats.csv")
        
        self._init_files()
        
        # 🆕 Conexión a Supabase
        self.use_database = use_database and SUPABASE_AVAILABLE
        self.db = None
        
        if self.use_database:
            try:
                self.db = ChessDatabase()
                print("GameLogger conectado a Supabase")
            except Exception as e:
                print(f"No se pudo conectar a Supabase: {e}")

                self.use_database = False
    
    def _init_files(self):

        
        # Archivo de movimientos
        if not os.path.exists(self.moves_file):
            with open(self.moves_file, 'w', newline='', encoding='utf-8') as f:
                writer.writerow([
                    'iteration', 'game_id', 'move_number', 'player', 
                    'move_algebraic_notation', 'move_confidence', 'board_fen'
                ])

        
        # Archivo de estadísticas
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'game_id', 'timestamp', 'total_moves',
                    'winner', 'termination_reason', 'unique_positions'
                ])
        
        # Archivo de datos de entrenamiento
        if not os.path.exists(self.training_file):
            with open(self.training_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'game_id', 'move_number', 'player',
                    'outcome', 'top_5_moves', 'top_5_probs', 'board_fen'
                ])

    def log_game_moves(self, iteration, game_id, moves_history):
        """
        moves_history: lista de tuplas con:
            (move_number, player, move_uci, move_confidence, board_fen)
        """
        with open(self.moves_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for move_data in moves_history:
                move_number, player, move_uci, move_confidence, board_fen = move_data

                try:
                    # Crear tablero desde el FEN antes del movimiento
                    board = chess.Board(board_fen)
                    
                    # Convertir de UCI → SAN (notación algebraica)
                    move = chess.Move.from_uci(move_uci)
                    move_san = board.san(move)
                except Exception as e:
                    # Si algo falla, guardamos el UCI como fallback
                    move_san = move_uci
                    print(f"⚠️ Error al convertir UCI '{move_uci}' a algebraico: {e}")

                writer.writerow([
                    iteration, game_id, move_number, player,
                    move_san, move_confidence, board_fen
                ])

    
    def log_game_stats(self, iteration, game_id, stats):
 
        # 1. Guardar en CSV (backup local)
        with open(self.stats_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                game_id,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                stats.get('total_moves', 0),
                stats.get('winner', 'draw'),
                stats.get('termination_reason', 'unknown'),
                stats.get('unique_positions', 0)
            ])
        
        if self.use_database and self.db:
            try:
                self.db.insert_game_stat(iteration, game_id, stats)
            except Exception as e:
                print(f"Error al guardar en Supabase: {e}")
                print(f"   (Datos guardados en CSV como backup)")
    
    def log_training_data(self, iteration, game_id, training_samples):

        with open(self.training_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in training_samples:
                move_num, player, policy, outcome, fen = sample
                
                # Obtener top 5 movimientos con mayor probabilidad
                top_5_indices = np.argsort(policy)[-5:][::-1]
                top_5_probs = policy[top_5_indices]
                
                # Convertir a strings para CSV
                top_moves_str = json.dumps(top_5_indices.tolist())
                top_probs_str = json.dumps(top_5_probs.tolist())
                
                writer.writerow([
                    iteration, game_id, move_num, player, outcome,
                    top_moves_str, top_probs_str, fen
                ])
    
    def get_game_summary(self, iteration):

        if self.use_database and self.db:
            try:
                summary = self.db.get_iteration_summary(iteration)
                if summary:
                    return summary
            except Exception as e:
                print(f"⚠️ Error al leer desde Supabase: {e}")
                print("   Leyendo desde CSV...")
        
        # Fallback: leer desde CSV
        return self._get_summary_from_csv(iteration)
    
    def _get_summary_from_csv(self, iteration):
        """Lee resumen desde archivo CSV local"""
        if not os.path.exists(self.stats_file):
            return {}
        
        white_wins = 0
        black_wins = 0
        draws = 0
        total_moves = []
        
        with open(self.stats_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['iteration']) == iteration:
                    winner = row['winner']
                    if winner == 'white':
                        white_wins += 1
                    elif winner == 'black':
                        black_wins += 1
                    else:
                        draws += 1
                    total_moves.append(int(row['total_moves']))
        
        total_games = white_wins + black_wins + draws
        
        return {
            'total_games': total_games,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
            'avg_moves': np.mean(total_moves) if total_moves else 0,
            'max_moves': max(total_moves) if total_moves else 0,
            'min_moves': min(total_moves) if total_moves else 0
        }
    
    def log_batch_stats(self, iteration, stats_list):

        with open(self.stats_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for game_id, stats in stats_list:
                writer.writerow([
                    iteration,
                    game_id,
                    timestamp,
                    stats.get('total_moves', 0),
                    stats.get('winner', 'draw'),
                    stats.get('termination_reason', 'unknown'),
                    stats.get('unique_positions', 0)
                ])
        
        # 2. 🆕 Guardar en Supabase en batch (mucho más rápido)
        if self.use_database and self.db:
            try:
                # Convertir a formato esperado por insert_game_stats_batch
                batch_data = [
                    (
                        iteration,
                        game_id,
                        stats.get('total_moves', 0),
                        stats.get('winner', 'draw'),
                        stats.get('termination_reason', 'unknown'),
                        stats.get('unique_positions', 0)
                    )
                    for game_id, stats in stats_list
                ]
                
                self.db.insert_game_stats_batch(batch_data)
                
            except Exception as e:
                print(f"Error al guardar batch en Supabase: {e}")
                print(f"   (Datos guardados en CSV como backup)")
    
    def close(self):
        """Cierra conexiones abiertas"""
        if self.use_database and self.db:
            try:
                self.db.close()
                print("Conexión a Supabase cerrada")
            except:
                pass



def format_winner(value, current_player_turn):

    if value == 0:
        return 'draw'
    elif value == 1:
        return 'white' if not current_player_turn else 'black'
    else:
        return 'black' if not current_player_turn else 'white'


def format_termination(state):

    if state.is_checkmate():
        return 'checkmate'
    elif state.is_stalemate():
        return 'stalemate'
    elif state.is_insufficient_material():
        return 'insufficient_material'
    elif state.is_seventyfive_moves():
        return 'seventy_five_moves'
    elif state.is_repetition(3):
        return 'threefold_repetition'
    else:
        return 'unknown'


