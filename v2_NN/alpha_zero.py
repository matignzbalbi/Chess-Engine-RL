import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import intel_extension_for_pytorch as ipex # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import random
from mcts import MCTS
from tqdm import tqdm
import os
import chess
from game_logger import GameLogger, format_winner, format_termination
import shutil
import tempfile
from pathlib import Path
import json
from datetime import datetime


# ==========================================
# 2. CLASE ALPHAZERO PRINCIPAL
# ==========================================
class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        # Configurar device (Intel XPU o CPU)
        if torch.xpu.is_available():
            self.device = torch.device("xpu")
            logging.info("Usando Intel GPU (XPU)")
        else:
            self.device = torch.device("cpu")
            logging.info("Intel GPU no disponible — usando CPU")

        self.model.to(self.device)

        # Optimización Intel IPEX (BFloat16 para Ponte Vecchio)
        try:
            self.model, self.optimizer = ipex.optimize(self.model, optimizer=self.optimizer, dtype=torch.float32)
            logging.info("Modelo optimizado con Intel Extension for PyTorch (IPEX)")
        except Exception as e:
            # ...
            logging.warning(f"No se pudo optimizar con IPEX: {e}")

        # LA CLAVE: Mover el estado del optimizador al dispositivo XPU/CPU
        # Esto corrige el error 'xpu:0 and cpu' en el optimizador.
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        logging.info(f"Usando device: {self.device}")

        # MCTS
        self.mcts = MCTS(game, args, model, device=self.device)

        # Logger
        self.logger = GameLogger()
        logging.info(f"Logs guardándose en: {self.logger.log_dir}/")
        
        # Configuración de directorio de checkpoints
        slurm_id = os.getenv("SLURM_JOB_ID")
        if slurm_id is not None:
            self.checkpoint_dir = f"pytorch_files_{slurm_id}"
        else:
            self.checkpoint_dir = "pytorch_files_local"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._verify_checkpoint_directory()

    # ------------------------------------------------------------------
    # UTILIDADES DE GUARDADO (Sin cambios mayores)
    # ------------------------------------------------------------------
    def _verify_checkpoint_directory(self):
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            test_file = os.path.join(self.checkpoint_dir, ".write_test")
            with open(test_file, 'w') as f: f.write("test")
            os.remove(test_file)
            logging.info(f" Directorio de checkpoints verificado: {self.checkpoint_dir}/")
        except Exception as e:
            logging.error(f"ERROR: Problema con directorio {self.checkpoint_dir}/: {e}")
            raise
    
    def _get_available_disk_space(self, path="."):
        try:
            return shutil.disk_usage(path).free / (1024 ** 2)
        except Exception:
            return float('inf')
    
    def _estimate_checkpoint_size(self):
        try:
            model_params = sum(p.numel() * 4 for p in self.model.parameters())
            return (model_params * 3) / (1024 ** 2) * 1.2 # Margen seguridad
        except Exception:
            return 100
    
    def _safe_save_checkpoint(self, obj, filepath, description="checkpoint"):
        if self._get_available_disk_space(self.checkpoint_dir) < self._estimate_checkpoint_size() * 2:
            logging.warning(f"POCO ESPACIO EN DISCO. No se guardó {description}")
            return False
        
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pt', dir=self.checkpoint_dir, prefix='.tmp_')
            os.close(temp_fd)
            torch.save(obj, temp_path)
            shutil.move(temp_path, filepath)
            return True
        except Exception as e:
            logging.error(f"ERROR guardando {description}: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path): os.remove(temp_path)
            return False

    def _save_checkpoint_bundle(self, iteration):
        success_count = 0
        base_name = f"model_{iteration}"
        logging.info(f"\nGuardando checkpoint de iteración {iteration}")
        
        model_path = os.path.join(self.checkpoint_dir, f"{base_name}.pt")
        optimizer_path = os.path.join(self.checkpoint_dir, f"{base_name}_optimizer.pt")
        config_path = os.path.join(self.checkpoint_dir, f"{base_name}_config.json")
        
        if self._safe_save_checkpoint(self.model.state_dict(), model_path, "modelo"): success_count += 1
        if self._safe_save_checkpoint(self.optimizer.state_dict(), optimizer_path, "optimizer"): success_count += 1
        
        try:
            config = {
                'iteration': iteration,
                'num_resBlocks': len(self.model.backBone),
                'num_hidden': self.model.startBlock[0].out_channels,
                'action_size': self.game.action_size,
                'timestamp': datetime.now().isoformat(),
                'args': self.args
            }
            with open(config_path, 'w') as f: json.dump(config, f, indent=2)
            success_count += 1
        except Exception: pass
        
        if success_count == 3:
            logging.info("Checkpoint completo guardado exitosamente")
            return True
        return False

    # ------------------------------------------------------------------
    # SELF PLAY (Corregido: Ruido y Recompensas)
    # ------------------------------------------------------------------
    def selfPlay(self, iteration=0, game_id=0):
        memory = []
        training_samples = []
        state = self.game.get_initial_state()
        move_count = 0
        play_history = []

        while True:
            action_probs = self.mcts.search(state)
            if not isinstance(action_probs, np.ndarray):
                action_probs = np.array(action_probs, dtype=np.float32)

            # Temperatura decreciente con el avance de la partida
            if move_count < 15:
                temperature = 1.0
            elif move_count < 40:
                temperature = 0.5
            else:
                temperature = 0.5 # Bajamos temperatura más tarde

            # Aplicar temperatura
            if temperature == 0:
                action = int(np.argmax(action_probs))
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    action_probs_temp = np.power(action_probs, 1.0 / temperature)
                if action_probs_temp.sum() > 0:
                    action_probs_temp /= action_probs_temp.sum()
                else:
                    action_probs_temp = action_probs / action_probs.sum()
                
                action = int(np.random.choice(self.game.action_size, p=action_probs_temp))

            play_history.append((state, action_probs.copy(), state.turn, action))
            
            state = self.game.get_next_state(state, action, 1)
            move_count += 1
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for idx, (hist_state, hist_action_probs, hist_turn, hist_chosen_action) in enumerate(play_history):
                    if value == 0:
                        # Empate - penalización ligera para incentivar decisión
                        hist_outcome = -0.1  # Penalización pequeña por empate
                    else:
                        winner_is_white = (state.turn == False)
                        is_winner = (hist_turn == winner_is_white)
                        
                        if is_winner:
                            # VICTORIA - recompensa con bonus por velocidad
                            base_reward = 1.0
                            
                            # Bonus por ganar rápido (máximo +30% si gana en <30 movimientos)
                            speed_bonus = max(0, (100 - move_count) / 100) * 0.3
                            
                            # Bonus por posición (más reward a movimientos finales que sellaron la victoria)
                            position_factor = (idx + 1) / len(play_history)  # 0.0 a 1.0
                            position_bonus = position_factor * 0.2  # Hasta +20% para moves finales
                            
                            hist_outcome = base_reward * (1 + speed_bonus + position_bonus)
                            hist_outcome = min(hist_outcome, 1.5)  # Cap máximo en 1.5
                        else:
                            # DERROTA - penalización más fuerte si perdió rápido
                            base_penalty = -1.0
                            
                            # Penalización extra por perder rápido
                            speed_penalty = max(0, (100 - move_count) / 100) * 0.3
                            
                            hist_outcome = base_penalty * (1 + speed_penalty)
                            hist_outcome = max(hist_outcome, -1.5)  # Cap mínimo en -1.5

                    encoded = self.game.get_encoded_state(hist_state)
                    returnMemory.append((encoded, hist_action_probs, hist_outcome))

                    try:
                        played_action = int(hist_chosen_action)
                        played_move = self.game.get_move_from_action(hist_state, played_action)
                        move_str = str(played_move)
                        played_conf = float(hist_action_probs[played_action])
                        
                        training_samples.append((
                            idx + 1, "white" if hist_turn else "black", move_str, 
                            f"{played_conf:.4f}", hist_action_probs, hist_outcome,
                            hist_state.fen() if hasattr(hist_state, "fen") else None
                        ))
                    except Exception: pass

                

                return returnMemory

    def train(self, memory):
        random.shuffle(memory)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            state_batch, policy_targets_batch, value_targets_batch = zip(*sample)

            state = torch.tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(np.array(policy_targets_batch), dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(np.array(value_targets_batch).reshape(-1, 1), dtype=torch.float32, device=self.device)

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

        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0

        return avg_policy_loss, avg_value_loss
    
    # ------------------------------------------------------------------
    # LEARN (Corregido: Logging y NaN check)
    # ------------------------------------------------------------------
    def learn(self):        
        SAVE_EVERY = self.args.get('save_every', 5)
        start_iter = self.args.get('start_iteration', 0)
        
        logging.info(f"CONFIGURACIÓN: Guardar cada {SAVE_EVERY} iteraciones")
        
        for iteration in range(start_iter, self.args['num_iterations']):
            logging.info(f"\n{'='*60}")
            logging.info(f"ITERACIÓN {iteration + 1}/{self.args['num_iterations']}")
            logging.info('='*60)

            memory = []
            self.model.eval()

            # Self Play
            logging.info(f"Generando datos ({self.args['num_selfPlay_iterations']} partidas)...")
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations']), desc="Self-play"):
                game_memory = self.selfPlay(iteration=iteration, game_id=selfPlay_iteration)
                memory += game_memory

            # Resumen breve
            summary = self.logger.get_game_summary(iteration)
            if summary:
                logging.info(f"Resumen: W:{summary.get('white_wins',0)} | B:{summary.get('black_wins',0)} | D:{summary.get('draws',0)}")

            # Entrenamiento
            self.model.train()
            logging.info(f"Entrenando modelo ({self.args['num_epochs']} épocas)...")

            for epoch in range(self.args['num_epochs']):
                p_loss, v_loss = self.train(memory)
                
                # Checkeo de seguridad NaN
                if np.isnan(p_loss) or np.isnan(v_loss):
                    logging.error("❌ ERROR: Loss es NaN. Deteniendo entrenamiento.")
                    return 

                # Imprimir siempre para ver progreso
                logging.info(f"  Época {epoch + 1}: P_Loss={p_loss:.4f} | V_Loss={v_loss:.4f}")

            # Guardado
            should_save = (iteration + 1) % SAVE_EVERY == 0 or iteration == self.args['num_iterations'] - 1

            if should_save:
                self._save_checkpoint_bundle(iteration)

            logging.info(f"Iteración {iteration + 1} completada")
        
        logging.info("ENTRENAMIENTO COMPLETADO")
        self._print_final_summary()
    
    def _print_final_summary(self):
        logging.info("\nModelos guardados:")
        model_files = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt') and not f.endswith('_optimizer.pt')])
        for model_file in model_files:
            path = os.path.join(self.checkpoint_dir, model_file)
            logging.info(f"  • {model_file} ({os.path.getsize(path)/(1024**2):.1f} MB)")