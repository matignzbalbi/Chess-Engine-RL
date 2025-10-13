import csv
import os
import numpy as np
from datetime import datetime
import json

class GameLogger:
    
    def __init__(self, log_dir="game_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Archivo para movimientos legibles
        self.moves_file = os.path.join(log_dir, "games_moves.csv")
        
        # Archivo para datos de entrenamiento
        self.training_file = os.path.join(log_dir, "training_data.csv")
        
        # Archivo para estadísticas de partidas
        self.stats_file = os.path.join(log_dir, "game_stats.csv")
        
        self._init_files()
    
    def _init_files(self):
        
        # Archivo de movimientos
        if not os.path.exists(self.moves_file):
            with open(self.moves_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'game_id', 'move_number', 'player', 
                    'move_uci', 'move_confidence', 'board_fen'
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

        with open(self.moves_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for move_data in moves_history:
                writer.writerow([iteration, game_id] + list(move_data))
    
    def log_game_stats(self, iteration, game_id, stats):
  
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
    
    
    
