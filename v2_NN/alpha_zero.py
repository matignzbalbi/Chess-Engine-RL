import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import random
from mcts import MCTS
from tqdm import tqdm
import os
import chess
from game_logger import GameLogger
import shutil
import tempfile
from pathlib import Path
import json
from datetime import datetime

# Importar IPEX al inicio
try:
    import intel_extension_for_pytorch as ipex # type: ignore
    HAS_IPEX = True
    logging.info(f"✓ Intel Extension for PyTorch {ipex.__version__} detectado")
except ImportError:
    HAS_IPEX = False
    logging.warning("⚠️ Intel Extension for PyTorch NO encontrado")

# Importar utilidades distribuidas
try:
    from ddp_utils import ( # type: ignore
        setup_distributed, 
        wrap_model_ddp, 
        is_main_process, 
        reduce_value, 
        get_rank, 
        barrier,
        cleanup_distributed
    )
    HAS_DDP_UTILS = True
except ImportError:
    HAS_DDP_UTILS = False
    logging.warning("ddp_utils.py no encontrado. Ejecutando en modo single-GPU")

class AlphaZero:
    def __init__(self, model, optimizer, game, args, rank=0, world_size=1):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # Setup distribuido si está disponible
        if HAS_DDP_UTILS:
            self.rank, self.world_size, self.local_rank = setup_distributed(backend='ccl')
            
            # Configurar device según rank
            if self.local_rank is not None and self.world_size > 1:
                self.device = torch.device(f'xpu:{self.local_rank}')
                self.device_type = 'xpu'
                logging.info(f"Proceso {self.rank}/{self.world_size}: usando XPU:{self.local_rank}")
            else:
                # Single GPU o no distribuido
                if HAS_IPEX and torch.xpu.is_available():
                    self.device = torch.device('xpu:0')
                    self.device_type = 'xpu'
                    logging.info(f"Modo single-GPU: XPU:0")
                else:
                    self.device = torch.device('cpu')
                    self.device_type = 'cpu'
                    logging.info(f"Modo CPU")
            
            # VERSIÓN SIMPLIFICADA: Solo mover a device, sin ipex.optimize()
            if self.device_type == 'xpu' and HAS_IPEX:
                try:
                    logging.info("Moviendo modelo a XPU con IPEX...")
                    
                    # Primero mover a XPU
                    self.model = self.model.to(self.device)
                    
                    # Intentar optimización simple (sin prepack)
                    # Solo para inference, no para training
                    # self.model = ipex.optimize(self.model, dtype=torch.float32, level="O0")
                    
                    logging.info("✓ Modelo en XPU")
                except Exception as e:
                    logging.error(f"❌ Error con XPU: {e}")
                    logging.error("Fallback a CPU")
                    self.device = torch.device('cpu')
                    self.device_type = 'cpu'
                    self.model = self.model.to(self.device)
            else:
                self.model = self.model.to(self.device)
            
            # Envolver con DDP si es multi-GPU
            if self.world_size > 1:
                self.model = wrap_model_ddp(self.model, self.device, self.device_type, self.local_rank)
                if is_main_process():
                    logging.info(f"✓ Entrenamiento distribuido: {self.world_size} GPUs Intel")
        else:
            # Fallback: sin DDP
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            
            # Verificar disponibilidad de XPU
            if HAS_IPEX and torch.xpu.is_available():
                self.device = torch.device('xpu:0')
                self.device_type = 'xpu'
                
                try:
                    logging.info("Moviendo modelo a XPU...")
                    self.model = self.model.to(self.device)
                    logging.info("✓ Modelo en XPU")
                except Exception as e:
                    logging.error(f"❌ Error con XPU: {e}")
                    logging.error("Fallback a CPU")
                    self.device = torch.device('cpu')
                    self.device_type = 'cpu'
                    self.model = self.model.to(self.device)
            else:
                self.device = torch.device('cpu')
                self.device_type = 'cpu'
                self.model = self.model.to(self.device)
            
            logging.info(f"✓ Usando device: {self.device}")

        # MCTS
        self.mcts = MCTS(game, args, self.model, device=self.device, device_type=self.device_type)

        # Logger (solo proceso principal)
        self.logger = GameLogger()
        if self._is_main():
            logging.info(f"Logs guardándose en: {self.logger.log_dir}/")
        
        self.checkpoint_dir = "pytorch_files"
        if self._is_main():
            self._verify_checkpoint_directory()

    def _is_main(self):
        """Helper: retorna True si es proceso principal"""
        if HAS_DDP_UTILS:
            return is_main_process()
        return True

    def _barrier(self):
        """Helper: sincroniza procesos si DDP está disponible"""
        if HAS_DDP_UTILS and self.world_size > 1:
            barrier()

    def _reduce_value(self, value, op='mean'):
        """Helper: reduce valor entre procesos si DDP está disponible"""
        if HAS_DDP_UTILS and self.world_size > 1:
            return reduce_value(value, op=op)
        return value

    def _verify_checkpoint_directory(self):
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            test_file = os.path.join(self.checkpoint_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            logging.info(f"✓ Directorio de checkpoints verificado: {self.checkpoint_dir}/")
        except PermissionError:
            logging.error(f"ERROR: Sin permisos de escritura en {self.checkpoint_dir}/")
            raise
        except OSError as e:
            logging.error(f"ERROR: No se puede crear directorio {self.checkpoint_dir}/: {e}")
            raise
    
    def _get_available_disk_space(self, path="."):
        try:
            stat = shutil.disk_usage(path)
            return stat.free / (1024 ** 2)
        except Exception:
            return float('inf')
    
    def _estimate_checkpoint_size(self):
        try:
            actual_model = self.model.module if hasattr(self.model, 'module') else self.model
            model_params = sum(p.numel() * 4 for p in actual_model.parameters())
            optimizer_params = model_params * 2
            total_bytes = model_params + optimizer_params
            total_mb = total_bytes / (1024 ** 2)
            return total_mb * 1.2
        except Exception:
            return 100
    
    def _safe_save_checkpoint(self, obj, filepath, description="checkpoint"):
        estimated_size = self._estimate_checkpoint_size()
        available_space = self._get_available_disk_space(self.checkpoint_dir)
        
        if available_space < estimated_size * 2:
            logging.info(f"ADVERTENCIA: Poco espacio en disco")
            logging.info(f"Disponible: {available_space:.1f} MB")
            logging.info(f"Necesario: {estimated_size * 2:.1f} MB")
            return False
        
        try:
            backup_path = None
            if os.path.exists(filepath):
                backup_path = filepath + ".backup"
                try:
                    shutil.copy2(filepath, backup_path)
                except Exception:
                    pass
            
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.pt',
                dir=self.checkpoint_dir,
                prefix='.tmp_'
            )
            os.close(temp_fd)
            
            # Mover a CPU antes de guardar
            if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
                obj_cpu = {k: v.cpu() for k, v in obj.items()}
                torch.save(obj_cpu, temp_path)
            else:
                torch.save(obj, temp_path)
            
            if not os.path.exists(temp_path):
                raise IOError(f"Archivo temporal {temp_path} no se creó")
            
            temp_size = os.path.getsize(temp_path)
            if temp_size < 1000:
                raise IOError(f"Archivo temporal muy pequeño: {temp_size} bytes")
            
            shutil.move(temp_path, filepath)
            
            if not os.path.exists(filepath):
                raise IOError(f"Archivo final {filepath} no existe")
            
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            logging.error(f"ERROR al guardar {description}: {type(e).__name__}: {e}")
            
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, filepath)
                    logging.info(f"Restaurado desde backup: {filepath}")
                except Exception:
                    pass
            
            return False
        
        finally:
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
    
    def _save_checkpoint_bundle(self, iteration):
        if not self._is_main():
            return True
        
        success_count = 0
        total_saves = 3
        
        base_name = f"model_{iteration}"
        logging.info(f"\nGuardando checkpoint de iteración {iteration}")
        
        model_path = os.path.join(self.checkpoint_dir, f"{base_name}.pt")
        optimizer_path = os.path.join(self.checkpoint_dir, f"{base_name}_optimizer.pt")
        config_path = os.path.join(self.checkpoint_dir, f"{base_name}_config.json")
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        logging.info(f"Guardando modelo...")
        if self._safe_save_checkpoint(model_to_save.state_dict(), model_path, f"modelo {base_name}"):
            size_mb = os.path.getsize(model_path) / (1024 ** 2)
            logging.info(f"✓ Modelo guardado ({size_mb:.1f} MB)")
            success_count += 1
        
        logging.info(f"Guardando optimizer...")
        if self._safe_save_checkpoint(self.optimizer.state_dict(), optimizer_path, f"optimizer {base_name}"):
            size_mb = os.path.getsize(optimizer_path) / (1024 ** 2)
            logging.info(f"✓ Optimizer guardado ({size_mb:.1f} MB)")
            success_count += 1
        
        logging.info(f"Guardando configuración...")
        try:
            config = {
                'iteration': iteration,
                'num_resBlocks': len(model_to_save.backBone), # type: ignore
                'num_hidden': model_to_save.startBlock[0].out_channels, # type: ignore
                'action_size': self.game.action_size,
                'training_games': (iteration + 1) * self.args['num_selfPlay_iterations'],
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'device_type': self.device_type,
                'pytorch_version': torch.__version__,
                'world_size': self.world_size,
                'args': self.args
            }
            
            if HAS_IPEX:
                config['ipex_version'] = ipex.__version__
            
            temp_config = config_path + ".tmp"
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            shutil.move(temp_config, config_path)
            logging.info(f"✓ Configuración guardada")
            success_count += 1
            
        except Exception as e:
            logging.info(f"❌ Falló configuración: {e}")
        
        if success_count == total_saves:
            logging.info(f"✅ Checkpoint completo")
            return True
        else:
            logging.info(f"⚠️ Checkpoint parcial ({success_count}/{total_saves})")
            return False

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

            if move_count < 15:
                temperature = 1.0
            elif move_count < 40:
                temperature = 0.5
            else:
                temperature = 0.1

            with np.errstate(divide='ignore', invalid='ignore'):
                action_probs_temp = np.power(action_probs, 1.0 / max(1e-8, temperature))

            if action_probs_temp.sum() > 0:
                action_probs_temp = action_probs_temp / float(action_probs_temp.sum())
            else:
                valid_moves_mask = self.game.get_valid_moves(state)
                valid_moves_mask = np.array(valid_moves_mask, dtype=np.float32)
                if valid_moves_mask.sum() > 0:
                    action_probs_temp = valid_moves_mask / float(valid_moves_mask.sum())
                else:
                    action_probs_temp = np.ones(self.game.action_size, dtype=np.float32)
                    action_probs_temp /= action_probs_temp.sum()

            action = int(np.random.choice(self.game.action_size, p=action_probs_temp))
            play_history.append((state, action_probs.copy(), state.turn, action))
            move = self.game.get_move_from_action(state, action)
            state = self.game.get_next_state(state, action, 1)
            move_count += 1
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for idx, (hist_state, hist_action_probs, hist_turn, hist_chosen_action) in enumerate(play_history):
                    if value == 0:
                        hist_outcome = 0
                    else:
                        winner_is_white = (state.turn == False)
                        if hist_turn == winner_is_white:
                            hist_outcome = 1
                        else:
                            hist_outcome = -1

                    encoded = self.game.get_encoded_state(hist_state)
                    returnMemory.append((encoded, hist_action_probs, hist_outcome))

                    try:
                        played_action = int(hist_chosen_action)
                    except Exception:
                        played_action = int(np.argmax(hist_action_probs))

                    played_move = self.game.get_move_from_action(hist_state, played_action)
                    move_uci = None
                    try:
                        if isinstance(played_move, chess.Move):
                            move_uci = played_move.uci()
                        else:
                            move_uci = str(played_move)
                    except Exception:
                        move_uci = str(played_move)

                    played_confidence = float(hist_action_probs[played_action]) if played_action < len(hist_action_probs) else 0.0
                    board_fen = hist_state.fen() if hasattr(hist_state, "fen") else None

                    training_samples.append((
                        idx + 1,
                        "white" if hist_turn else "black",
                        move_uci,
                        f"{played_confidence:.4f}",
                        hist_action_probs,
                        hist_outcome,
                        board_fen
                    ))

                if self._is_main():
                    try:
                        from game_logger import format_termination
                        self.logger.log_training_data(iteration, game_id, training_samples, self.game) # type: ignore
                        
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
                        self.logger.log_game_stats(iteration, game_id, stats) # type: ignore
                    except Exception as e:
                        logging.error(f"Error guardando logs: {e}")

                return returnMemory

    def train(self, memory):
        random.shuffle(memory)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            state_batch, policy_targets_batch, value_targets_batch = zip(*sample)

            state = torch.tensor(np.array(state_batch), dtype=torch.float32).to(self.device)
            policy_targets = torch.tensor(np.array(policy_targets_batch), dtype=torch.float32).to(self.device)
            value_targets = torch.tensor(np.array(value_targets_batch).reshape(-1, 1), dtype=torch.float32).to(self.device)

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
        
        avg_policy_loss = self._reduce_value(avg_policy_loss, op='mean')
        avg_value_loss = self._reduce_value(avg_value_loss, op='mean')

        return avg_policy_loss, avg_value_loss

    def learn(self):
        SAVE_EVERY = self.args.get('save_every', 5)
        start_iter = self.args.get('start_iteration', 0)
        
        if self._is_main():
            logging.info(f"\n{'='*70}")
            logging.info(f"CONFIGURACIÓN DE ENTRENAMIENTO")
            logging.info(f"{'='*70}")
            logging.info(f"Device: {self.device} ({self.device_type})")
            logging.info(f"Procesos: {self.world_size}")
            logging.info(f"Checkpoint cada {SAVE_EVERY} iteraciones")
            logging.info(f"{'='*70}\n")
        
        for iteration in range(start_iter, self.args['num_iterations']):
            if self._is_main():
                logging.info(f"\n{'='*60}")
                logging.info(f"ITERACIÓN {iteration + 1}/{self.args['num_iterations']}")
                logging.info('='*60)

            memory = []
            self.model.eval()

            if self._is_main():
                logging.info(f"\nSelf-play: {self.args['num_selfPlay_iterations']} partidas")
            
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                game_memory = self.selfPlay(iteration=iteration, game_id=selfPlay_iteration)
                memory += game_memory

            self._barrier()

            if self._is_main():
                logging.info(f"✓ Estados generados: {len(memory)}")

                summary = self.logger.get_game_summary(iteration)
                if summary:
                    logging.info(f"\nResumen:")
                    logging.info(f"   Blancas: {summary.get('white_wins', 0)} | "
                          f"Negras: {summary.get('black_wins', 0)} | "
                          f"Empates: {summary.get('draws', 0)}")

            self.model.train()
            
            if self._is_main():
                logging.info(f"\nEntrenando: {self.args['num_epochs']} épocas")

            for epoch in range(self.args['num_epochs']):
                avg_policy_loss, avg_value_loss = self.train(memory)
                
                if self._is_main() and ((epoch + 1) % 10 == 0 or epoch == 0):
                    print(f"   Época {epoch + 1}: Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")

            self._barrier()

            should_save = (iteration + 1) % SAVE_EVERY == 0 or iteration == self.args['num_iterations'] - 1
            
            if should_save:
                self._save_checkpoint_bundle(iteration)
            
            self._barrier()
            
            if self._is_main():
                logging.info(f"\n✓ Iteración {iteration + 1} completada")
        
        if self._is_main():
            logging.info("\n" + "="*70)
            logging.info("✅ ENTRENAMIENTO COMPLETADO")
            logging.info("="*70)
        
        if HAS_DDP_UTILS and self.world_size > 1:
            cleanup_distributed()
    
    def _print_final_summary(self):
        logging.info("\nModelos guardados:")
        
        model_files = sorted([f for f in os.listdir(self.checkpoint_dir) 
                            if f.endswith('.pt') and not f.endswith('_optimizer.pt')])
        
        if not model_files:
            logging.info("No se encontraron modelos guardados")
            return
        
        total_size = 0
        for model_file in model_files:
            path = os.path.join(self.checkpoint_dir, model_file)
            size_mb = os.path.getsize(path) / (1024**2)
            total_size += size_mb
            logging.info(f"   • {model_file:30} ({size_mb:.1f} MB)")
        
        logging.info(f"\n   Total: {len(model_files)} modelos, {total_size:.1f} MB")