import numpy as np
import torch
from mcts import MCTS
from chess_game import ChessGame
from model import create_chess_model


def play_game_with_neural_mcts():
    
    
    print("=== INICIALIZANDO SISTEMA ===\n")
    
    # Configuración de MCTS
    args = {
        'C': 2.0,  # Constante de exploración (más alta que antes porque usamos priors)
        'num_searches': 200 # Menos búsquedas necesarias con la red neuronal
    }
    
    # Inicializar juego
    game = ChessGame()
    
    # Crear modelo (pequeño para pruebas)
    print("Creando modelo de red neuronal...")
    model = create_chess_model(
        game=game,
        num_resBlocks=4,   # Pequeño para testing
        num_hidden=64      # Pequeño para testing
    )
    model.eval()  # Modo evaluación (sin dropout, etc.)
    print("✓ Modelo creado (NO entrenado, predicciones aleatorias)\n")
    
    # Crear MCTS con modelo
    mcts = MCTS(game, args, model)
    
    # Estado inicial
    state = game.get_initial_state()
    
    move_count = 0
    max_moves = 100  # Limitar para testing
    
    print("=== INICIANDO PARTIDA ===\n")
    game.render(state)
    
    while move_count < max_moves:
        # Verificar si el juego terminó
        value, is_terminal = game.get_value_and_terminated(state, None)
        
        if is_terminal:
            print(f"\n{'='*50}")
            print("JUEGO TERMINADO")
            print('='*50)
            if value == 1:
                print("🏆 Ganaron las BLANCAS")
            elif value == -1:
                print("🏆 Ganaron las NEGRAS")
            else:
                print("🤝 EMPATE")
            break
        
        # Obtener jugador actual
        player = "Blancas ♔" if state.turn else "Negras ♚"
        print(f"\n{'='*50}")
        print(f"Turno {move_count + 1} - {player}")
        print('='*50)
        
        # Ejecutar MCTS
        print(f"Ejecutando MCTS ({args['num_searches']} simulaciones)...")
        action_probs = mcts.search(state)
        
        # Obtener top 3 movimientos considerados
        legal_moves = list(state.legal_moves)
        top_k = min(3, len(legal_moves))
        top_indices = np.argsort(action_probs)[-top_k:][::-1]
        
        print("\nTop movimientos considerados:")
        for i, idx in enumerate(top_indices):
            if idx < len(legal_moves):
                move = legal_moves[idx]
                prob = action_probs[idx]
                print(f"  {i+1}. {move.uci()} - Visitas: {prob:.1%}")
        
        # Seleccionar mejor movimiento
        action = np.argmax(action_probs)
        move = game.get_move_from_action(state, action)
        
        print(f"\n→ Movimiento seleccionado: {move.uci()}")
        print(f"  Confianza: {action_probs[action]:.1%}")
        
        # Aplicar movimiento
        state = game.get_next_state(state, action, 1)
        
        print()
        game.render(state)
        move_count += 1
    
    if move_count >= max_moves:
        print(f"\n{'='*50}")
        print(f"JUEGO TERMINADO POR LÍMITE ({max_moves} movimientos)")
        print('='*50)
    
    print(f"\nTotal de movimientos: {move_count}")


def compare_mcts_with_and_without_model():
    
    print("\n" + "="*60)
    print("=== COMPARACIÓN: ROLLOUTS vs RED NEURONAL ===")
    print("="*60 + "\n")
    
    game = ChessGame()
    state = game.get_initial_state()
    
    # Hacer algunos movimientos para tener una posición interesante
    state.push_uci("e2e4")
    state.push_uci("e7e5")
    state.push_uci("g1f3")
    
    print("Posición de prueba:")
    game.render(state)
    
    # Evaluar con modelo
    print("\n--- Con Red Neuronal ---")
    model = create_chess_model(game, num_resBlocks=2, num_hidden=32)
    model.eval()
    
    encoded = game.get_encoded_state(state)
    state_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        policy_logits, value = model(state_tensor)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
    
    print(f"Value del modelo: {value.item():.4f}")
    print(f"Número de movimientos con prob > 0.01: {np.sum(policy > 0.01)}")
    
    # Mostrar distribución del policy
    valid_moves = game.get_valid_moves(state)
    legal_policy = policy * valid_moves
    legal_policy /= legal_policy.sum()
    
    legal_moves = list(state.legal_moves)
    top_5 = np.argsort(legal_policy)[-5:][::-1]
    
    print("\nTop 5 movimientos según el modelo:")
    for i, idx in enumerate(top_5):
        if idx < len(legal_moves):
            print(f"  {i+1}. {legal_moves[idx].uci()} - {legal_policy[idx]:.2%}")


if __name__ == "__main__":
    # Descomentar la que quieras ejecutar:
    
    play_game_with_neural_mcts()
    
    
    # compare_mcts_with_and_without_model()