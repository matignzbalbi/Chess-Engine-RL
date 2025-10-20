import numpy as np
import torch
import torch.nn.functional as F
import random
from mcts import MCTS
from tqdm import tqdm
import os
from game_logger import GameLogger, format_winner, format_termination

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        # Configurar device y mover modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Usando device: {self.device}")

        # Crear MCTS (pasando modelo ya en GPU)
        self.mcts = MCTS(game, args, model, device=self.device) # type: ignore

        # Inicializar logger
        self.logger = GameLogger()
        print(f"📊 Logs guardándose en: {self.logger.log_dir}/")

    def selfPlay(self, iteration=0, game_id=0):
        memory = []
        moves_history = []
        training_samples = []

        state = self.game.get_initial_state()
        move_count = 0

        while True:
            # Ejecutar MCTS para obtener policy
            action_probs = self.mcts.search(state)

            memory.append((state.copy(), action_probs, state.turn))

            # Temperatura según move_count
            if move_count < 15:
                temperature = 1.0
            elif move_count < 40:
                temperature = 0.5
            else:
                temperature = 0.1

            action_probs_temp = action_probs ** (1 / temperature)
            action_probs_temp /= action_probs_temp.sum()

            action = np.random.choice(self.game.action_size, p=action_probs_temp)

            # Registrar movimiento
            move = self.game.get_move_from_action(state, action)
            player = "white" if state.turn else "black"
            confidence = action_probs[action]
            fen = state.fen()
            moves_history.append((move_count + 1, player, move.uci(), f"{confidence:.4f}", fen))

            # Aplicar movimiento
            state = self.game.get_next_state(state, action, 1)
            move_count += 1

            # Verificar fin de juego
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if is_terminal:
                returnMemory = []
                for idx, (hist_state, hist_action_probs, hist_turn) in enumerate(memory):
                    hist_outcome = 0 if value == 0 else (value if hist_turn != state.turn else -value)
                    returnMemory.append((self.game.get_encoded_state(hist_state), hist_action_probs, hist_outcome))
                    training_samples.append((idx + 1, "white" if hist_turn else "black", hist_action_probs, hist_outcome, hist_state.fen()))

                # Guardar logs
                self.logger.log_game_moves(iteration, game_id, moves_history)
                self.logger.log_training_data(iteration, game_id, training_samples)
                winner = format_winner(value, state.turn)
                termination = format_termination(state)
                stats = {'total_moves': move_count, 'winner': winner, 'termination_reason': termination,
                         'unique_positions': len(set(m[4] for m in moves_history))}
                self.logger.log_game_stats(iteration, game_id, stats)

                return returnMemory

    def train(self, memory):
        random.shuffle(memory)

        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)

            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.device)

            out_policy, out_value = self.model(state)
            policy_loss = -torch.sum(policy_targets * F.log_softmax(out_policy, dim=1)) / policy_targets.size(0)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

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
        for iteration in range(self.args['num_iterations']):
            print(f"\n{'='*60}")
            print(f"ITERACIÓN {iteration + 1}/{self.args['num_iterations']}")
            print('='*60)

            memory = []
            self.model.eval()
            print(f"\nGenerando datos con self-play ({self.args['num_selfPlay_iterations']} partidas)...")
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations']), desc="Self-play"):
                game_memory = self.selfPlay(iteration=iteration, game_id=selfPlay_iteration)
                memory += game_memory

            print(f"✓ Generados {len(memory)} estados de entrenamiento")

            summary = self.logger.get_game_summary(iteration)
            if summary:
                print(f"\n📊 Resumen de partidas:")
                print(f"   Blancas: {summary['white_wins']} | Negras: {summary['black_wins']} | Empates: {summary['draws']}")
                print(f"   Promedio de movimientos: {summary['avg_moves']:.1f}")

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
            torch.save(self.model.state_dict(), f"pytorch_files/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"pytorch_files/optimizer_{iteration}.pt")
            print(f"✓ Iteración {iteration + 1} completada")
