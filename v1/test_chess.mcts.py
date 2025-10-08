import numpy as np
from mcts import MCTS
from chess_game import ChessGame

def play_game():
    
    """Juega una partida de ajedrez usando MCTS"""
    
    # Configuración de MCTS
    args = {
        'C': 1.41,  # Constante de exploración (sqrt(2))
        'num_searches': 100  # Número de simulaciones por movimiento
    }
    
    # Inicializar juego y MCTS
    game = ChessGame()
    mcts = MCTS(game, args)
    
    # Estado inicial
    state = game.get_initial_state()
    
    move_count = 0
    max_moves = 200  # Limitar partida para testing
    
    print("=== INICIANDO PARTIDA DE AJEDREZ CON MCTS ===\n")
    game.render(state)
    
    while move_count < max_moves:
        # Verificar si el juego terminó
        value, is_terminal = game.get_value_and_terminated(state, None)
        
        if is_terminal:
            print(f"\n=== JUEGO TERMINADO ===")
            if value == 1:
                print("Ganaron las BLANCAS")
            elif value == -1:
                print("Ganaron las NEGRAS")
            else:
                print("EMPATE")
            break
        
        # Obtener jugador actual
        player = "Blancas" if state.turn else "Negras"
        print(f"\nTurno {move_count + 1} - Jugador: {player}")
        print("Ejecutando MCTS...")
        
        # Ejecutar MCTS para obtener mejor movimiento
        action_probs = mcts.search(state)
        
        # Seleccionar acción con mayor probabilidad
        action = np.argmax(action_probs)
        
        # Obtener el movimiento de ajedrez
        move = game.get_move_from_action(state, action)
        print(f"Movimiento seleccionado: {move}")
        print(f"Confianza: {action_probs[action]:.3f}")
        
        # Aplicar movimiento
        state = game.get_next_state(state, action, 1)
        
        game.render(state)
        move_count += 1
    
    if move_count >= max_moves:
        print(f"\n=== JUEGO TERMINADO POR LÍMITE DE MOVIMIENTOS ({max_moves}) ===")
    
    print(f"\nTotal de movimientos: {move_count}")

if __name__ == "__main__":
    play_game()