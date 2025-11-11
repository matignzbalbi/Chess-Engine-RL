# evaluate_checkpoints.py
# Versión adaptada a la API de tu repo v2_NN (ChessGame, MCTS, model.create_chess_model)
import os
import glob
import torch
from pathlib import Path
from rating_manager import RatingManager
from chess_game import ChessGame
from model import create_chess_model, ChessResNet
from mcts import MCTS
import argparse
import random
import numpy as np

CHECKPOINT_DIR = "pytorch_files"
GAMES_PER_PAIR = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_path(path, game):
    """
    Carga modelo intentando detectar num_hidden automáticamente desde checkpoint,
    o usando archivo *_config.json si existe.
    """
    import json
    default_config = {"num_resBlocks": 5, "num_hidden": 64}
    config_path = os.path.splitext(path)[0] + "_config.json"
    sd = torch.load(path, map_location=torch.device("cpu"))

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        num_resBlocks = config.get("num_resBlocks", default_config["num_resBlocks"])
        num_hidden = config.get("num_hidden", default_config["num_hidden"])
        print(f"[INFO] Config encontrada para {path}: {config}")
    else:
        # Detectar num_hidden mirando startBlock.0.weight si existe
        detected_hidden = None
        for k, v in sd.items():
            if "startBlock.0.weight" in k or "startBlock.0.conv" in k:
                detected_hidden = v.shape[0]
                break
        if detected_hidden:
            num_hidden = detected_hidden
            print(f"[INFO] Detectado automáticamente num_hidden={num_hidden} desde {path}")
        else:
            num_hidden = default_config["num_hidden"]
            print(f"[WARN] No detectado num_hidden en {path}. Usando {num_hidden}")
        num_resBlocks = default_config["num_resBlocks"]

    # Crear el modelo con la firma que usa tu repo
    try:
        model = ChessResNet(game, num_resBlocks, num_hidden)
    except Exception:
        # Si utilizás una fábrica en model.py (create_chess_model) úsala:
        model = create_chess_model(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)

    # Intentar cargar
    try:
        model.load_state_dict(sd)
        print(f"[OK] Modelo cargado completamente desde {path}")
    except RuntimeError as e:
        print(f"[WARN] Error al cargar completamente {path}: {str(e).splitlines()[0]}")
        model.load_state_dict(sd, strict=False)
        print(f"[DONE] Modelo cargado parcialmente desde {path}")

    model.to(DEVICE)
    model.eval()
    return model

def play_one_game(modelA, modelB, game, mcts_args_a, mcts_args_b, max_moves=400):
    """
    Juega una partida entre modelA (white) y modelB (black).
    Devuelve 1.0 si gana A (white), 0.5 empate, 0.0 si gana B.
    """
    state = game.get_initial_state()  # tu método real
    move_count = 0

    # Alternaremos turnos consultando game.get_value_and_terminated(state)
    while True:
        value, terminated = game.get_value_and_terminated(state)
        if terminated:
            # value: -1 si mate (desde la perspectiva del jugador que movió?), 
            # en tu implementación: -1 para checkmate => probablemente el jugador que acaba de mover perdió.
            # En la práctica, lo que nos interesa es interpretar winner:
            # Para simplicidad: llamamos a game.get_value_and_terminated y si terminated -> revisar winner con is_checkmate/stalemate
            break
        # Decidir qué modelo juega ahora: en la implementación del state (python-chess) .turn indica True=white
        # Pero tu state es chess.Board() y tiene attribute .turn
        is_white_to_move = state.turn
        if is_white_to_move:
            mcts = MCTS(game, mcts_args_a, modelA, device=DEVICE)
            action_probs = mcts.search(state)
        else:
            mcts = MCTS(game, mcts_args_b, modelB, device=DEVICE)
            action_probs = mcts.search(state)

        # seleccionar acción determinísticamente (puedes cambiar a muestreo)
        if np.sum(action_probs) == 0:
            # fallback: elegir acción legal al azar
            legal_mask = game.get_valid_moves(state)
            legal_idx = np.flatnonzero(legal_mask)
            if len(legal_idx) == 0:
                # Sin movimientos -> terminar
                break
            action = int(np.random.choice(legal_idx))
        else:
            action = int(np.argmax(action_probs))

        # aplicar acción
        # get_next_state(state, action, player) en tu código requiere player, pasamos 1 como en tu test_play.py
        state = game.get_next_state(state, action, 1)
        move_count += 1
        if move_count >= max_moves:
            break

    # Interpretar resultado usando get_value_and_terminated OR usando inspección final:
    val, terminated = game.get_value_and_terminated(state)
    # Si tu función devuelve -1 para checkmate (probablemente signifique el jugador que movió perdió)
    # Lo más robusto: inspeccionar estado final con python-chess:
    # state.is_checkmate() / state.is_stalemate()
    if state.is_checkmate():
        # quien está en turno perdió -> si turn is True (white to move) -> black delivered mate -> black wins
        if state.turn:
            # white to move but checkmate -> black won -> return 0.0
            return 0.0
        else:
            return 1.0
    elif state.is_stalemate() or state.is_insufficient_material() or state.is_fivefold_repetition() or state.is_seventyfive_moves():
        return 0.5
    else:
        # fallback: usar value returned
        try:
            # Si value is from perspective of current player, no trivial mapping; fallback a empate
            return 0.5
        except Exception:
            return 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_pair", type=int, default=GAMES_PER_PAIR)
    parser.add_argument("--use_glicko", action="store_true")
    args = parser.parse_args()

    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "model_*.pt")))
    if len(ckpts) < 2:
        print("Necesitás al menos 2 checkpoints en", CHECKPOINT_DIR)
        return

    game = ChessGame()
    rating = RatingManager(use_glicko=args.use_glicko, storage_path="ratings.csv")

    for ck in ckpts:
        rating.ensure_player(os.path.basename(ck))

    # Cargar modelos en memoria
    models = {}
    for ck in ckpts:
        models[ck] = load_model_from_path(ck, game)

    mcts_args = {"num_searches": 2, "C": 1.0}



    for i in range(len(ckpts)):
        for j in range(i+1, len(ckpts)):
            a = ckpts[i]; b = ckpts[j]
            a_name = os.path.basename(a); b_name = os.path.basename(b)
            wins_a = draws = wins_b = 0
            for k in range(args.per_pair):
                # alternar colores
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
            print(f"Match {a_name} vs {b_name}: A_wins={wins_a}, draws={draws}, B_wins={wins_b}")
    rating.save()
    print("Ratings saved to ratings.csv")

if __name__ == "__main__":
    main()
