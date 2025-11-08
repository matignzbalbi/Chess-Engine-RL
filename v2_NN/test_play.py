import torch
import numpy as np
from chess_game import ChessGame
from model import create_chess_model
from mcts import MCTS

# Cargar modelo
game = ChessGame()
model = create_chess_model(game, num_resBlocks=2, num_hidden=32)
model.load_state_dict(torch.load("pytorch_files\model_4.pt"))  
model.eval()

# Crear MCTS
args = {'C': 2, 'num_searches': 100}
mcts = MCTS(game, args, model)

state = game.get_initial_state()
move_num = 0

while True:
    move_num += 1
    print(f"\n--- Movimiento {move_num} ---")
    game.render(state)
    
    value, is_terminal = game.get_value_and_terminated(state, None)
    if is_terminal:
        print("JUEGO TERMINADO")
        if value == 1:
            print("Ganador:", "Blancas" if not state.turn else "Negras")
        else:
            print("Empate")
        break
    
    USE_TEMPERATURE = True
    temperature = 0.7

    action_probs = mcts.search(state)

    if USE_TEMPERATURE:
        action_probs_temp = action_probs ** (1 / temperature)
        action_probs_temp /= action_probs_temp.sum()
        action = np.random.choice(len(action_probs), p=action_probs_temp)
        print(action_probs)
        print(len(action_probs))
    else:
        action = np.argmax(action_probs)

    move = game.get_move_from_action(state, action)
    print(f"{'Blancas' if state.turn else 'Negras'} juega: {move.uci()}")
    state = game.get_next_state(state, action, 1) 