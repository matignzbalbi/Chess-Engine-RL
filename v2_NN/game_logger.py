import csv
import os
import numpy as np
from datetime import datetime
import chess


class GameLogger:

    def __init__(self, log_dir: str = "game_logs") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Archivos CSV (almacenamiento local)
        self.training_file = os.path.join(log_dir, "training_data.csv")
        self.stats_file = os.path.join(log_dir, "game_stats.csv")

        self._init_files()

    # ---------------------------------------------------------------------
    def _init_files(self) -> None:

        # Archivo de estadísticas
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iteration", "game_id", "timestamp", "total_moves",
                    "winner", "termination_reason", "unique_positions"
                ])

        # Archivo de datos de entrenamiento
        if not os.path.exists(self.training_file):
            with open(self.training_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iteration", "game_id", "move_number", "player",
                    "move_algebraic_notation", "move_confidence",
                    "outcome", "top_5_moves", "top_5_probs", "board_fen"
                ])

    # ---------------------------------------------------------------------
    def log_game_stats(self, iteration: int, game_id: str, stats: dict) -> None:

        # Guardar en CSV
        with open(self.stats_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                game_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                stats.get("total_moves", 0),
                stats.get("winner", "draw"),
                stats.get("termination_reason", "unknown"),
                stats.get("unique_positions", 0)
            ])

    # ---------------------------------------------------------------------
    def log_training_data(self, iteration: int, game_id: str, training_samples: list, game_instance) -> None:
     
        with open(self.training_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for sample in training_samples:
                move_num, player, move_uci, move_confidence, policy, outcome, fen = sample

                # Convertir de UCI a SAN (notación algebraica)
                try:
                    board = chess.Board(fen, chess960=False)
                    move = chess.Move.from_uci(move_uci)
                    move_san = board.san(move)
                except Exception as e:
                    move_san = move_uci
                    print(f"⚠️ Error al convertir '{move_uci}' a SAN: {e}")

                # Obtener top 5 movimientos
                top_5_indices = np.argsort(policy)[-5:][::-1]
                top_5_probs = policy[top_5_indices]

                # Convertir índices a SAN
                top_5_moves = []
                for action_idx in top_5_indices:
                    try:
                        action_move = game_instance.get_move_from_action(board, action_idx)
                        top_5_moves.append(board.san(action_move) if action_move else f"action_{action_idx}")
                    except Exception:
                        top_5_moves.append(f"action_{action_idx}")

                top_moves_str = ", ".join(top_5_moves)
                top_probs_str = ", ".join([f"{p:.4f}" for p in top_5_probs])

                writer.writerow([
                    iteration, game_id, move_num, player,
                    move_san, move_confidence,
                    outcome, top_moves_str, top_probs_str, fen
                ])

    # ---------------------------------------------------------------------
    def get_game_summary(self, iteration: int) -> dict:
        return self._get_summary_from_csv(iteration)

    # ---------------------------------------------------------------------
    def _get_summary_from_csv(self, iteration: int) -> dict:
        if not os.path.exists(self.stats_file):
            return {}

        white_wins, black_wins, draws = 0, 0, 0
        total_moves = []

        with open(self.stats_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["iteration"]) == iteration:
                    winner = row["winner"]
                    if winner == "white":
                        white_wins += 1
                    elif winner == "black":
                        black_wins += 1
                    else:
                        draws += 1
                    total_moves.append(int(row["total_moves"]))

        total_games = white_wins + black_wins + draws
        if not total_games:
            return {}

        return {
            "total_games": total_games,
            "white_wins": white_wins,
            "black_wins": black_wins,
            "draws": draws,
            "avg_moves": float(np.mean(total_moves)) if total_moves else 0,
            "max_moves": int(max(total_moves)) if total_moves else 0,
            "min_moves": int(min(total_moves)) if total_moves else 0,
        }

    # ---------------------------------------------------------------------
    def log_batch_stats(self, iteration: int, stats_list: list) -> None:
        if not stats_list:
            return

        # Guardar en CSV
        with open(self.stats_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for game_id, stats in stats_list:
                writer.writerow([
                    iteration,
                    game_id,
                    timestamp,
                    stats.get("total_moves", 0),
                    stats.get("winner", "draw"),
                    stats.get("termination_reason", "unknown"),
                    stats.get("unique_positions", 0),
                ])

    # ---------------------------------------------------------------------
    def close(self) -> None:
        """Cierra conexiones abiertas (no hace nada sin base de datos)."""
        pass

def format_winner(value: int, current_player_turn: bool) -> str:
    """Formatea el ganador según value y turno actual"""
    if value == 0:
        return "draw"
    elif value == 1:
        return "white" if not current_player_turn else "black"
    else:
        return "black" if not current_player_turn else "white"


def format_termination(state: chess.Board) -> str:
    if state.is_checkmate():
        return "checkmate"
    elif state.is_stalemate():
        return "stalemate"
    elif state.is_insufficient_material():
        return "insufficient_material"
    elif state.is_seventyfive_moves():
        return "seventy_five_moves"
    elif state.is_repetition(3):
        return "threefold_repetition"
    else:
        return "unknown"