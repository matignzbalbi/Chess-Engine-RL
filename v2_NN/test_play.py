import torch
import numpy as np
from chess_game import ChessGame
from model import create_chess_model
from mcts import MCTS

# Cargar modelo
game = ChessGame()
model = create_chess_model(game, num_resBlocks=4, num_hidden=64)
model.load_state_dict(torch.load("pytorch_files\model_1.pt"))  
model.eval()

# Crear MCTS
args = {'C': 2, 'num_searches': 100}
mcts = MCTS(game, args, model)


# Jugar
state = game.get_initial_state()

for move_num in range(1, 300):  # Max 100 movimientos
    print(f"\n--- Movimiento {move_num} ---")
    game.render(state)
    
    # Verificar si terminó
    value, is_terminal = game.get_value_and_terminated(state, None)
    if is_terminal:
        print("JUEGO TERMINADO")
        if value == 1:
            print("Ganador:", "Blancas" if not state.turn else "Negras")
        else:
            print("Empate")
        break
    
    # IA juega
    action_probs = mcts.search(state)
    action = np.argmax(action_probs)
    move = game.get_move_from_action(state, action)
    print(f"{'Blancas' if state.turn else 'Negras'} juega: {move.uci()}")
    
    state = game.get_next_state(state, action, 1)
    
    ### Con temperatura ###
    
    """   USE_TEMPERATURE = True
    temperature = 0.7

    # IA juega
    action_probs = mcts.search(state)

    if USE_TEMPERATURE:
        # Con variedad
        action_probs_temp = action_probs ** (1 / temperature)
        action_probs_temp /= action_probs_temp.sum()
        action = np.random.choice(len(action_probs), p=action_probs_temp)
    else:
        # Determinista (como antes)
        action = np.argmax(action_probs)

    move = game.get_move_from_action(state, action)
    print(f"{'Blancas' if state.turn else 'Negras'} juega: {move.uci()}")
    state = game.get_next_state(state, action, 1) """