import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
import csv
import math
import glob
import torch
import random
import numpy as np
import argparse
from pathlib import Path

# ----------------------------------------------------------------------
#  RATING MANAGER (antes rating_manager.py)
# ----------------------------------------------------------------------

class SimpleEloPlayer:
    def __init__(self, rating=1500):
        self.rating = rating
        self.games = 0

def expected_score_elo(Ra, Rb):
    return 1 / (1 + 10 ** ((Rb - Ra) / 400))

class RatingManager:
    def __init__(self, use_glicko=False, storage_path="ratings.csv"):
        self.use_glicko = use_glicko
        self.storage_path = storage_path
        self.players = {}   # name -> SimpleEloPlayer
        self.load()

    # ------------------------------------------------------------
    #  RATING STORAGE
    # ------------------------------------------------------------
    def load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", newline='', encoding="utf-8") as f:
                r = csv.reader(f)
                next(r)
                for row in r:
                    name, rating, games = row
                    self.players[name] = SimpleEloPlayer(float(rating))
                    self.players[name].games = int(games)
        except Exception as e:
            logging.error(f"Error loading ratings: {e}")

    def save(self):
        try:
            with open(self.storage_path, "w", newline='', encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["name", "rating", "games"])
                for name, p in self.players.items():
                    w.writerow([name, p.rating, p.games])
        except Exception as e:
            logging.error(f"Error saving ratings: {e}")

    # ------------------------------------------------------------
    #  OPERACIONES SOBRE RATINGS
    # ------------------------------------------------------------
    def ensure_player(self, name):
        if name not in self.players:
            self.players[name] = SimpleEloPlayer()

    def record_match_result(self, a, b, a_result):
        """
        a_result:
            1.0 = A gana
            0.0 = B gana
            0.5 = tablas
        """
        self.ensure_player(a)
        self.ensure_player(b)

        Pa = expected_score_elo(self.players[a].rating, self.players[b].rating)
        Pb = 1 - Pa

        K = 20

        self.players[a].rating += K * (a_result - Pa)
        self.players[b].rating += K * ((1 - a_result) - Pb)

        self.players[a].games += 1
        self.players[b].games += 1

# ----------------------------------------------------------------------
#  EVALUACIÓN CHECKPOINTS
# ----------------------------------------------------------------------

from chess_game import ChessGame  
from model import create_chess_model, ChessResNet
from mcts import MCTS

CHECKPOINT_DIR = "pytorch_files"
GAMES_PER_PAIR = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
#   CARGAR MODELO DESDE .PT
# ------------------------------------------------------------
def load_model_from_path(path, game):
    import json

    default_config = {"num_resBlocks": 5, "num_hidden": 64}
    config_path = os.path.splitext(path)[0] + "_config.json"

    sd = torch.load(path, map_location=torch.device("cpu"))

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        num_resBlocks = config.get("num_resBlocks", default_config["num_resBlocks"])
        num_hidden = config.get("num_hidden", default_config["num_hidden"])
        logging.info(f"Config encontrada para {path}: {config}")
    else:
        detected_hidden = None
        for k, v in sd.items():
            if "startBlock.0.weight" in k or "startBlock.0.conv" in k:
                detected_hidden = v.shape[0]
                break
        num_hidden = detected_hidden if detected_hidden else default_config["num_hidden"]
        num_resBlocks = default_config["num_resBlocks"]

    try:
        model = ChessResNet(game, num_resBlocks, num_hidden)
    except Exception:
        model = create_chess_model(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)

    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        logging.error(f"Error parcial al cargar {path}: {e}")
        model.load_state_dict(sd, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

# ------------------------------------------------------------
#   JUGAR UNA PARTIDA ENTRE DOS MODELOS
# ------------------------------------------------------------
def play_one_game(modelA, modelB, game, mcts_args_a, mcts_args_b, max_moves=400):

    state = game.get_initial_state()
    move_count = 0

    while True:
        value, terminated = game.get_value_and_terminated(state)
        if terminated:
            break

        is_white_to_move = state.turn
        if is_white_to_move:
            mcts = MCTS(game, mcts_args_a, modelA, device=DEVICE)
            action_probs = mcts.search(state)
        else:
            mcts = MCTS(game, mcts_args_b, modelB, device=DEVICE)
            action_probs = mcts.search(state)

        if np.sum(action_probs) == 0:
            legal_mask = game.get_valid_moves(state)
            legal_idx = np.flatnonzero(legal_mask)
            if len(legal_idx) == 0:
                break
            action = int(np.random.choice(legal_idx))
        else:
            action = int(np.argmax(action_probs))

        state = game.get_next_state(state, action, 1)
        move_count += 1
        if move_count >= max_moves:
            break

    val, terminated = game.get_value_and_terminated(state)

    if state.is_checkmate():
        return 1.0 if not state.turn else 0.0
    elif state.is_stalemate() or state.is_insufficient_material() or state.is_fivefold_repetition() or state.is_fifty_moves():
        return 0.5
    return 0.5

# ------------------------------------------------------------
#   MAIN (UNIFICADO)
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_pair", type=int, default=GAMES_PER_PAIR)
    args = parser.parse_args()

    ckpts = sorted([
        f for f in glob.glob(os.path.join(CHECKPOINT_DIR, "model_*.pt"))
        if "optimizer" not in f and "config" not in f
    ])
    if len(ckpts) < 2:
        logging.info("Necesitás al menos 2 checkpoints en pytorch_files/")
        return

    game = ChessGame()
    # Crear carpeta de logs si no existe
    os.makedirs("game_logs", exist_ok=True)
    rating = RatingManager(storage_path="game_logs/ratings.csv")

    for ck in ckpts:
        rating.ensure_player(os.path.basename(ck))

    # Cargar modelos
    models = {ck: load_model_from_path(ck, game) for ck in ckpts}

    mcts_args = {"num_searches": 20, "C": 6.0}

    # Torneo Round Robin
    for i in range(len(ckpts)):
        for j in range(i+1, len(ckpts)):
            a = ckpts[i]
            b = ckpts[j]
            a_name = os.path.basename(a)
            b_name = os.path.basename(b)

            wins_a = draws = wins_b = 0

            for k in range(args.per_pair):
                if k % 2 == 0:
                    res = play_one_game(models[a], models[b], game, mcts_args, mcts_args)
                else:
                    r = play_one_game(models[b], models[a], game, mcts_args, mcts_args)
                    res = 1.0 - r

                if res == 1.0:
                    wins_a += 1
                elif res == 0.5:
                    draws += 1
                else:
                    wins_b += 1

                rating.record_match_result(a_name, b_name, res)

            logging.info(f"{a_name} vs {b_name} — A:{wins_a}  D:{draws}  B:{wins_b}")

    rating.save()
    logging.info("Ratings guardados en ratings.csv")


if __name__ == "__main__":
    main()