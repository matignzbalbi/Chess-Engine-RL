import numpy as np
import torch
import torch.nn.functional as F
import random
from mcts import MCTS
from tqdm import tqdm
import os
from game_logger import GameLogger, format_winner, format_termination

class AlphaZero:
    """
    Implementación de AlphaZero para ajedrez.
    
    CONVENCIONES DE VALORES (CRÍTICO):
    - Durante self-play, guardamos (state, policy, turno)
    - Al terminar, calculamos el outcome desde la perspectiva de cada estado
    - La red aprende a predecir valores desde la perspectiva del jugador actual
    """
    
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        # Configurar device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Usando device: {self.device}")

        # Crear MCTS (pasando modelo ya en GPU/CPU)
        self.mcts = MCTS(game, args, model, device=self.device)

        # Inicializar logger
        self.logger = GameLogger()
        print(f"📊 Logs guardándose en: {self.logger.log_dir}/")

    def selfPlay(self, iteration=0, game_id=0):
        """
        Ejecuta una partida de self-play.
        
        FLUJO DE VALORES:
        1. Guardamos (state, policy, turno_del_jugador) en cada movimiento
        2. Al final, calculamos outcome desde perspectiva de cada jugador
        3. Si el jugador en ese turno ganó → outcome = +1
        4. Si el jugador en ese turno perdió → outcome = -1
        5. Si empate → outcome = 0
        
        Returns:
            Lista de (encoded_state, policy, outcome) para entrenamiento
        """
        memory = []
        moves_history = []
        training_samples = []

        state = self.game.get_initial_state()
        move_count = 0

        while True:
            # Ejecutar MCTS para obtener policy
            action_probs = self.mcts.search(state)

            # Guardar: (estado, policy, turno del jugador que movió)
            # state.turn es True para blancas, False para negras
            memory.append((state.copy(), action_probs, state.turn))

            # Aplicar temperatura para exploración
            if move_count < 15:
                temperature = 1.0
            elif move_count < 40:
                temperature = 0.5
            else:
                temperature = 0.1

            # Samplear acción con temperatura
            action_probs_temp = action_probs ** (1 / temperature)
            if action_probs_temp.sum() > 0:
                action_probs_temp /= action_probs_temp.sum()
            else:
                # Fallback: uniforme sobre movimientos válidos
                valid_moves = self.game.get_valid_moves(state)
                action_probs_temp = valid_moves / valid_moves.sum()

            action = np.random.choice(self.game.action_size, p=action_probs_temp)

            # Registrar movimiento para logs
            move = self.game.get_move_from_action(state, action)
            player_name = "white" if state.turn else "black"
            confidence = action_probs[action]
            fen = state.fen()
            moves_history.append((
                move_count + 1, 
                player_name, 
                move.uci(), 
                f"{confidence:.4f}", 
                fen
            ))

            # Aplicar movimiento
            state = self.game.get_next_state(state, action, 1)
            move_count += 1

            # Verificar fin de juego
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                # ✅ CORREGIDO: Calcular outcomes correctamente
                #
                # 'value' está desde la perspectiva del jugador ACTUAL (state.turn)
                # Pero queremos saber quién GANÓ la partida
                #
                # Si value = -1: El jugador actual (state.turn) perdió
                #               → El oponente (quien movió último) ganó
                # Si value = 0: Empate
                
                returnMemory = []
                
                for idx, (hist_state, hist_action_probs, hist_turn) in enumerate(memory):
                    # Determinar el outcome desde la perspectiva del jugador en hist_turn
                    
                    if value == 0:
                        # Empate: outcome = 0 para todos
                        hist_outcome = 0
                    else:
                        # Alguien ganó
                        # value = -1 significa que state.turn perdió
                        # Entonces !state.turn ganó
                        
                        winner_is_white = (state.turn == False)  # Si negras tienen turno, blancas ganaron
                        
                        if hist_turn == winner_is_white:
                            # Este jugador ganó
                            hist_outcome = 1
                        else:
                            # Este jugador perdió
                            hist_outcome = -1
                    
                    # Agregar a memoria de entrenamiento
                    encoded = self.game.get_encoded_state(hist_state)
                    returnMemory.append((encoded, hist_action_probs, hist_outcome))
                    
                    # Agregar a training samples para logs
                    training_samples.append((
                        idx + 1,
                        "white" if hist_turn else "black",
                        hist_action_probs,
                        hist_outcome,
                        hist_state.fen()
                    ))

                # Guardar logs
                self.logger.log_game_moves(iteration, game_id, moves_history)
                self.logger.log_training_data(iteration, game_id, training_samples)
                
                # Determinar ganador para estadísticas
                if value == 0:
                    winner = 'draw'
                else:
                    winner = 'white' if (state.turn == False) else 'black'
                
                termination = format_termination(state)
                stats = {
                    'total_moves': move_count,
                    'winner': winner,
                    'termination_reason': termination,
                    'unique_positions': len(set(m[4] for m in moves_history))
                }
                self.logger.log_game_stats(iteration, game_id, stats)

                return returnMemory

    def train(self, memory):
        """
        Entrena el modelo con los datos de self-play.
        
        Args:
            memory: Lista de (encoded_state, policy_target, value_target)
        
        Returns:
            (avg_policy_loss, avg_value_loss): Pérdidas promedio
        """
        random.shuffle(memory)

        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)

            # Convertir a tensores
            state = torch.tensor(
                np.array(state), 
                dtype=torch.float32, 
                device=self.device
            )
            policy_targets = torch.tensor(
                np.array(policy_targets), 
                dtype=torch.float32, 
                device=self.device
            )
            value_targets = torch.tensor(
                np.array(value_targets).reshape(-1, 1), 
                dtype=torch.float32, 
                device=self.device
            )

            # Forward pass
            out_policy, out_value = self.model(state)
            
            # Loss de policy: Cross-entropy entre target y predicción
            policy_loss = -torch.sum(
                policy_targets * F.log_softmax(out_policy, dim=1)
            ) / policy_targets.size(0)
            
            # Loss de value: MSE entre target y predicción
            value_loss = F.mse_loss(out_value, value_targets)
            
            # Loss total
            loss = policy_loss + value_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0
        
        return avg_policy_loss, avg_value_loss

    def learn(self):
        """
        Ciclo principal de entrenamiento AlphaZero.
        
        Para cada iteración:
        1. Self-play: Generar datos jugando contra sí mismo
        2. Train: Entrenar la red con los datos generados
        3. Save: Guardar el modelo
        """
        for iteration in range(self.args['num_iterations']):
            print(f"\n{'='*60}")
            print(f"ITERACIÓN {iteration + 1}/{self.args['num_iterations']}")
            print('='*60)

            memory = []
            self.model.eval()
            
            print(f"\nGenerando datos con self-play ({self.args['num_selfPlay_iterations']} partidas)...")
            for selfPlay_iteration in tqdm(
                range(self.args['num_selfPlay_iterations']), 
                desc="Self-play"
            ):
                game_memory = self.selfPlay(
                    iteration=iteration, 
                    game_id=selfPlay_iteration
                )
                memory += game_memory

            print(f"✓ Generados {len(memory)} estados de entrenamiento")

            # Mostrar resumen de partidas
            summary = self.logger.get_game_summary(iteration)
            if summary:
                print(f"\n📊 Resumen de partidas:")
                print(f"   Blancas: {summary['white_wins']} | "
                      f"Negras: {summary['black_wins']} | "
                      f"Empates: {summary['draws']}")
                print(f"   Promedio de movimientos: {summary['avg_moves']:.1f}")

            # Entrenar modelo
            self.model.train()
            print(f"\nEntrenando modelo ({self.args['num_epochs']} épocas)...")
            
            for epoch in range(self.args['num_epochs']):
                avg_policy_loss, avg_value_loss = self.train(memory)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Época {epoch + 1}/{self.args['num_epochs']}: "
                          f"Policy Loss = {avg_policy_loss:.4f}, "
                          f"Value Loss = {avg_value_loss:.4f}")

            # Guardar modelo
            print(f"\n✓ Guardando modelo de iteración {iteration}...")
            os.makedirs("pytorch_files", exist_ok=True)
            torch.save(
                self.model.state_dict(), 
                f"pytorch_files/model_{iteration}.pt"
            )
            torch.save(
                self.optimizer.state_dict(), 
                f"pytorch_files/optimizer_{iteration}.pt"
            )
            print(f"✓ Iteración {iteration + 1} completada")


# === TEST ===

def test_value_calculation():
    """
    Test para verificar que los valores se calculan correctamente en self-play.
    """
    import chess
    from chess_game import ChessGame
    from model import create_chess_model
    
    print("=" * 70)
    print("TEST: CÁLCULO DE VALORES EN SELF-PLAY")
    print("=" * 70)
    
    # Setup
    game = ChessGame(include_queen_promotions=False)
    model = create_chess_model(game, num_resBlocks=2, num_hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    args = {
        'C': 2,
        'num_searches': 10,
        'num_iterations': 1,
        'num_selfPlay_iterations': 1,
        'num_epochs': 1,
        'batch_size': 16
    }
    
    az = AlphaZero(model, optimizer, game, args)
    
    print("\n1️⃣ Simulando partida donde ganan blancas...")
    
    # Simular memoria de una partida
    memory = [
        (game.get_initial_state(), np.random.rand(game.action_size), True),   # Blancas
        (game.get_initial_state(), np.random.rand(game.action_size), False),  # Negras
        (game.get_initial_state(), np.random.rand(game.action_size), True),   # Blancas
    ]
    
    # Simular final donde ganan blancas
    # state.turn = False (turno de negras), value = -1 (negras perdieron)
    # Entonces blancas ganaron
    
    state_final = chess.Board()
    value = -1  # Jugador actual (quien tenga turno) perdió
    state_final._turn = False  # type: ignore # Turno de negras
    
    print(f"  Estado final: turno={'Blancas' if state_final.turn else 'Negras'}, value={value}")
    print(f"  Ganador: Blancas")
    
    # Calcular outcomes
    winner_is_white = (state_final.turn == False)  # True
    
    print(f"\n  Outcomes esperados:")
    for idx, (_, _, hist_turn) in enumerate(memory):
        player = "Blancas" if hist_turn else "Negras"
        expected_outcome = 1 if hist_turn == winner_is_white else -1
        print(f"    Movimiento {idx+1} ({player}): {expected_outcome:+d}")
    
    print("\n  ✓ Blancas (movimientos 1 y 3) deben tener outcome = +1")
    print("  ✓ Negras (movimiento 2) debe tener outcome = -1")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    test_value_calculation()