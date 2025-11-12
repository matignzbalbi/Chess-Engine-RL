import numpy as np
import torch
import torch.nn.functional as F
import random
from mcts import MCTS
from tqdm import tqdm
import os
import chess
from game_logger import GameLogger, format_winner, format_termination

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        # Configurar device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Usando device: {self.device}")

        # MCTS (se asume que MCTS acepta device)
        self.mcts = MCTS(game, args, model, device=self.device)

        # Logger
        self.logger = GameLogger()
        print(f"Logs guardándose en: {self.logger.log_dir}/")

    def selfPlay(self, iteration=0, game_id=0):
        """
        Ejecuta una partida por self-play y devuelve la memoria (lista de tuples)
        Cada entrada en memory_return será: (encoded_state, policy, outcome)
        """
        memory = []  # Para construir training returnMemory
        training_samples = []  # Para enviar al GameLogger (move_uci, policy, fen, etc.)

        state = self.game.get_initial_state()
        move_count = 0

        # Nota: ahora guardamos también la acción elegida en el historial para reconstrucción exacta
        play_history = []  # lista de (state_copy, action_probs, turn, chosen_action)

        while True:
            # Ejecutar MCTS para obtener policy (numpy array length action_size)
            action_probs = self.mcts.search(state)  # se asume numpy array
            if not isinstance(action_probs, np.ndarray):
                action_probs = np.array(action_probs, dtype=np.float32)

            # Guardar estado + política + turno (y luego la acción elegida)
            # Usamos state.copy() si tu objeto state lo soporta; si no, guardá una representación (p.ej. FEN)
            # Para mantener compatibilidad con GameLogger, no convertimos aquí a FEN; lo haremos cuando armemos training_samples
            # Pero guardamos el state entero para usar game.get_move_from_action(board, action) a la hora de reconstruir.
            # Agregaremos chosen_action tras seleccionar la acción.
            # Aplicar temperatura para exploración
            if move_count < 15:
                temperature = 1.0
            elif move_count < 40:
                temperature = 0.5
            else:
                temperature = 0.1

            # Aplicar temperatura (exponente sobre probabilidades)
            # Evitar división por cero
            with np.errstate(divide='ignore', invalid='ignore'):
                action_probs_temp = np.power(action_probs, 1.0 / max(1e-8, temperature))

            if action_probs_temp.sum() > 0:
                action_probs_temp = action_probs_temp / float(action_probs_temp.sum())
            else:
                # Fallback: uniforme sobre movimientos válidos (se asume array binaria o mask con probs)
                valid_moves_mask = self.game.get_valid_moves(state)  # se asume numpy array del mismo largo
                valid_moves_mask = np.array(valid_moves_mask, dtype=np.float32)
                if valid_moves_mask.sum() > 0:
                    action_probs_temp = valid_moves_mask / float(valid_moves_mask.sum())
                else:
                    # Como último recurso, distribución uniforme completa
                    action_probs_temp = np.ones(self.game.action_size, dtype=np.float32)
                    action_probs_temp /= action_probs_temp.sum()

            # Selección de acción
            action = int(np.random.choice(self.game.action_size, p=action_probs_temp))

            # Registrar en historial (añadiremos la action escogida)
            play_history.append((state, action_probs.copy(), state.turn, action))

            # Obtener movimiento (puede devolver chess.Move o string UCI) y confianza
            move = self.game.get_move_from_action(state, action)
            confidence = float(action_probs[action]) if action < len(action_probs) else 0.0
            fen = state.fen() if hasattr(state, "fen") else None

            # Aplicar movimiento y avanzar estado
            state = self.game.get_next_state(state, action, 1)
            move_count += 1

            # Verificar fin de juego (value + is_terminal)
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                # Construimos returnMemory y training_samples usando play_history
                returnMemory = []

                for idx, (hist_state, hist_action_probs, hist_turn, hist_chosen_action) in enumerate(play_history):
                    # Determinar outcome según 'value' (mantengo la lógica original)
                    if value == 0:
                        hist_outcome = 0
                    else:
                        # Si en el estado terminal "state.turn == False" entonces (según lógica original)
                        # winner_is_white = (state.turn == False)
                        winner_is_white = (state.turn == False)
                        # hist_turn es True si en ese hist_state era turno de blancas
                        if hist_turn == winner_is_white:
                            hist_outcome = 1
                        else:
                            hist_outcome = -1

                    # Encoded state para entrenamiento (vector/array)
                    encoded = self.game.get_encoded_state(hist_state)
                    returnMemory.append((encoded, hist_action_probs, hist_outcome))

                    # Reconstruimos el movimiento realmente jugado en ese hist_state:
                    # Usamos hist_chosen_action (la acción que realmente se escogió)
                    try:
                        played_action = int(hist_chosen_action)
                    except Exception:
                        # fallback: tomar argmax de la política si algo raro pasó
                        played_action = int(np.argmax(hist_action_probs))

                    # Obtener el movimiento desde la acción (puede ser chess.Move o string)
                    played_move = self.game.get_move_from_action(hist_state, played_action)
                    # Normalizar move a string UCI si es posible
                    move_uci = None
                    try:
                        if isinstance(played_move, chess.Move):
                            move_uci = played_move.uci()
                        elif isinstance(played_move, str):
                            move_uci = played_move  # asumimos UCI o SAN; el logger tiene lógica robusta para interpretar
                        else:
                            # intentar str()
                            move_uci = str(played_move)
                    except Exception:
                        move_uci = str(played_move)

                    played_confidence = float(hist_action_probs[played_action]) if played_action < len(hist_action_probs) else 0.0
                    board_fen = hist_state.fen() if hasattr(hist_state, "fen") else None

                    # Agregamos sample en el formato que espera GameLogger:
                    # (move_number, player, move_uci, move_confidence, policy, outcome, fen)
                    training_samples.append((
                        idx + 1,  # move_number (1-based)
                        "white" if hist_turn else "black",
                        move_uci,
                        f"{played_confidence:.4f}",
                        hist_action_probs,
                        hist_outcome,
                        board_fen
                    ))

                # Guardar training data consolidado
                try:
                    self.logger.log_training_data(iteration, game_id, training_samples, self.game) # type: ignore
                except Exception as e:
                    print(f"⚠️ Error guardando training data: {e}")

                # Determinar ganador para estadísticas (mantengo la lógica previa)
                if value == 0:
                    winner = 'draw'
                else:
                    winner = 'white' if (state.turn == False) else 'black'

                termination = format_termination(state)
                stats = {
                    'total_moves': move_count,
                    'winner': winner,
                    'termination_reason': termination,
                    'unique_positions': len(set(s[6] for s in training_samples if s[6] is not None))
                }
                # Guardar stats
                try:
                    self.logger.log_game_stats(iteration, game_id, stats) # type: ignore
                except Exception as e:
                    print(f"⚠️ Error guardando stats de la partida: {e}")

                # Retornar memoria para entrenamiento (encoded, policy, outcome)
                return returnMemory

    def train(self, memory):
        """Entrena la red a partir de memory = [(state_encoded, policy, value), ...]"""
        random.shuffle(memory)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            state_batch, policy_targets_batch, value_targets_batch = zip(*sample)

            # Convertir a tensores
            state = torch.tensor(
                np.array(state_batch),
                dtype=torch.float32,
                device=self.device
            )
            policy_targets = torch.tensor(
                np.array(policy_targets_batch),
                dtype=torch.float32,
                device=self.device
            )
            value_targets = torch.tensor(
                np.array(value_targets_batch).reshape(-1, 1),
                dtype=torch.float32,
                device=self.device
            )

            out_policy, out_value = self.model(state)

            # Policy loss: cross-entropy (targets are distributions)
            policy_loss = -torch.sum(policy_targets * F.log_softmax(out_policy, dim=1)) / policy_targets.size(0)

            # Value loss: MSE
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0

        return avg_policy_loss, avg_value_loss

    def learn(self):
        """Loop principal de aprendizaje (iteraciones de self-play + entrenamiento)"""
        for iteration in range(self.args['num_iterations']):
            print(f"\n{'='*60}")
            print(f"ITERACIÓN {iteration + 1}/{self.args['num_iterations']}")
            print('='*60)

            memory = []
            self.model.eval()

            print(f"\nGenerando datos con self-play ({self.args['num_selfPlay_iterations']} partidas)")
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations']), desc="Self-play"):
                game_memory = self.selfPlay(iteration=iteration, game_id=selfPlay_iteration)
                # game_memory es una lista de (encoded_state, policy, outcome)
                memory += game_memory

            print(f"Generados {len(memory)} estados de entrenamiento")

            # Mostrar resumen de partidas desde logger
            summary = self.logger.get_game_summary(iteration)
            if summary:
                print(f"\nResumen de partidas:")
                print(f"Blancas: {summary.get('white_wins', 0)} | "
                      f"Negras: {summary.get('black_wins', 0)} | "
                      f"Empates: {summary.get('draws', 0)}")
                if 'avg_moves' in summary:
                    try:
                        print(f"Promedio de movimientos: {summary['avg_moves']:.1f}")
                    except Exception:
                        print("Promedio de movimientos: (no disponible)")
                else:
                    print("⚠️ No hay datos para calcular el promedio de movimientos.")

            # Entrenar modelo
            self.model.train()
            print(f"\nEntrenando modelo ({self.args['num_epochs']} épocas)")

            for epoch in range(self.args['num_epochs']):
                avg_policy_loss, avg_value_loss = self.train(memory)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Época {epoch + 1}/{self.args['num_epochs']}: "
                          f"Policy Loss = {avg_policy_loss:.4f}, "
                          f"Value Loss = {avg_value_loss:.4f}")

            # Guardar modelo/optimizer
            print(f"\nGuardando modelo de iteración {iteration}")
            os.makedirs("pytorch_files", exist_ok=True)
            torch.save(self.model.state_dict(), f"pytorch_files/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"pytorch_files/optimizer_{iteration}.pt")
            print(f"Iteración {iteration + 1} completada")
