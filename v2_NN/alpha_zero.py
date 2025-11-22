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
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    def __init__(self, memory):
        self.memory = memory

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        state, policy_targets, value_targets = self.memory[idx]
        
        # Convertimos a tipos correctos aquí para que el DataLoader 
        # pueda armar los tensores automáticamente de forma eficiente.
        return (
            np.array(state, dtype=np.float32), 
            np.array(policy_targets, dtype=np.float32), 
            np.array(value_targets, dtype=np.float32)
        )

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        # Configurar device
        # Intel GPU (XPU) support with fallback to CPU
        if torch.xpu.is_available():
            self.device = torch.device("xpu")
            logging.info("Usando Intel GPU (XPU)")
        else:
            self.device = torch.device("cpu")
            logging.info("Intel GPU no disponible — usando CPU")

        self.model.to(self.device)

        # Optimización Intel IPEX
        try:
            self.model, self.optimizer = ipex.optimize(self.model, optimizer=self.optimizer, dtype=torch.bfloat16)
            logging.info("Modelo optimizado con Intel Extension for PyTorch (IPEX)")
        except Exception as e:
            logging.warning(f"No se pudo optimizar con IPEX: {e}")

        logging.info(f"Usando device: {self.device}")

        # MCTS
        self.mcts = MCTS(game, args, model, device=self.device)

        # Logger
        self.logger = GameLogger()
        logging.info(f"Logs guardándose en: {self.logger.log_dir}/")
        
        # Verificar directorio de checkpoints al inicio
        slurm_id = os.getenv("SLURM_JOB_ID")

        if slurm_id is not None:
            # Carpeta única por job
            self.checkpoint_dir = f"pytorch_files_{slurm_id}"
        else:
            # Fallback cuando lo ejecutás a mano
            self.checkpoint_dir = "pytorch_files_local"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._verify_checkpoint_directory()


    def _verify_checkpoint_directory(self):
        try:
            # Crear directorio si no existe
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Verificar escritura con archivo temporal
            test_file = os.path.join(self.checkpoint_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            logging.info(f" Directorio de checkpoints verificado: {self.checkpoint_dir}/")
            
        except PermissionError:
            logging.error(f"ERROR: Sin permisos de escritura en {self.checkpoint_dir}/")
            raise
        except OSError as e:
            logging.error(f"ERROR: No se puede crear directorio {self.checkpoint_dir}/: {e}")
            raise
    
    def _get_available_disk_space(self, path="."):
        """Obtiene el espacio disponible en disco en MB"""
        try:
            stat = shutil.disk_usage(path)
            return stat.free / (1024 ** 2)  # Convertir a MB
        except Exception:
            return float('inf')  # Si falla, asumir espacio infinito
    
    def _estimate_checkpoint_size(self):
        try:
            # Calcular tamaño del modelo
            model_params = sum(p.numel() * 4 for p in self.model.parameters())  # 4 bytes por float32
            
            # Calcular tamaño del optimizer (aproximado, depende del tipo)
            optimizer_params = model_params * 2  # Adam guarda 2 momentos por parámetro
            
            total_bytes = model_params + optimizer_params
            total_mb = total_bytes / (1024 ** 2)
            
            # Agregar 20% de margen para metadatos y JSON
            return total_mb * 1.2
            
        except Exception:
            return 100  # Fallback conservador: 100 MB
    
    def _safe_save_checkpoint(self, obj, filepath, description="checkpoint"):
        # Verificar espacio en disco
        estimated_size = self._estimate_checkpoint_size()
        available_space = self._get_available_disk_space(self.checkpoint_dir)
        
        if available_space < estimated_size * 2:  # Requerir 2x el tamaño (seguridad)
            logging.info(f"ADVERTENCIA: Poco espacio en disco")
            logging.info(f"Disponible: {available_space:.1f} MB")
            logging.info(f"Necesario: {estimated_size * 2:.1f} MB")
            logging.info(f"NO se guardó {description}")
            return False
        
        try:
            # Crear backup si el archivo ya existe
            backup_path = None
            if os.path.exists(filepath):
                backup_path = filepath + ".backup"
                try:
                    shutil.copy2(filepath, backup_path)
                except Exception as e:
                    logging.info(f"No se pudo crear backup de {filepath}: {e}")
            
            # Guardar en archivo temporal (guardado atómico)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.pt',
                dir=self.checkpoint_dir,
                prefix='.tmp_'
            )
            os.close(temp_fd)
            
            # Guardar en temporal
            torch.save(obj, temp_path)
            
            # Verificar que el archivo temporal se escribió correctamente
            if not os.path.exists(temp_path):
                raise IOError(f"Archivo temporal {temp_path} no se creó")
            
            temp_size = os.path.getsize(temp_path)
            if temp_size < 1000:  # Menos de 1KB es sospechoso
                raise IOError(f"Archivo temporal muy pequeño: {temp_size} bytes")
            
            # Mover archivo temporal al destino final (operación atómica)
            shutil.move(temp_path, filepath)
            
            # Verificar que el archivo final existe
            if not os.path.exists(filepath):
                raise IOError(f"Archivo final {filepath} no existe después de mover")
            
            # Eliminar backup si todo salió bien
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
            
            return True
            
        except IOError as e:
            logging.error(f"ERROR de E/S al guardar {description}: {e}")
            
            # Restaurar desde backup si existe
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, filepath)
                    logging.info(f"Restaurado desde backup: {filepath}")
                except Exception as restore_err:
                    logging.info(f"No se pudo restaurar backup: {restore_err}")
            
            return False
            
        except RuntimeError as e:
            logging.error(f"ERROR de PyTorch al guardar {description}: {e}")
            return False
            
        except MemoryError:
            logging.error(f"ERROR: Sin memoria para guardar {description}")
            return False
            
        except Exception as e:
            logging.error(f"ERROR inesperado al guardar {description}: {type(e).__name__}: {e}")
            return False
        
        finally:
            # Limpiar archivos temporales si quedaron
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
    
    def _save_checkpoint_bundle(self, iteration):
        success_count = 0
        total_saves = 3  # modelo + optimizer + config
        
        base_name = f"model_{iteration}"
        logging.info(f"\nGuardando checkpoint de iteración {iteration}")
        
        model_path = os.path.join(self.checkpoint_dir, f"{base_name}.pt")
        optimizer_path = os.path.join(self.checkpoint_dir, f"{base_name}_optimizer.pt")
        config_path = os.path.join(self.checkpoint_dir, f"{base_name}_config.json")
        
        # Guardar modelo
        logging.info(f"Guardando modelo...")
        if self._safe_save_checkpoint(self.model.state_dict(), model_path, f"modelo {base_name}"):
            size_mb = os.path.getsize(model_path) / (1024 ** 2)
            logging.info(f"Modelo guardado ({size_mb:.1f} MB)")
            success_count += 1
        else:
            logging.info(f"Falló guardado de modelo")
        
        # Guardar optimizer
        logging.info(f"Guardando optimizer...")
        if self._safe_save_checkpoint(self.optimizer.state_dict(), optimizer_path, f"optimizer {base_name}"):
            size_mb = os.path.getsize(optimizer_path) / (1024 ** 2)
            logging.info(f" Optimizer guardado ({size_mb:.1f} MB)")
            success_count += 1
        else:
            logging.info(f"Falló guardado de optimizer")
        
        logging.info(f"Guardando configuración")
        try:
            config = {
                'iteration': iteration,
                'num_resBlocks': len(self.model.backBone),
                'num_hidden': self.model.startBlock[0].out_channels,
                'action_size': self.game.action_size,
                'training_games': (iteration + 1) * self.args['num_selfPlay_iterations'],
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'args': self.args
            }
            
            temp_config = config_path + ".tmp"
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            shutil.move(temp_config, config_path)
            logging.info(f" Configuración guardada")
            success_count += 1
            
        except Exception as e:
            logging.info(f"Falló guardado de configuración: {e}")
        
        # Resultado final
        if success_count == total_saves:
            logging.info(f"Checkpoint completo guardado exitosamente")
            return True
        elif success_count > 0:
            logging.info(f"Checkpoint guardado parcialmente ({success_count}/{total_saves})")
            return False
        else:
            logging.info(f"Falló completamente el guardado del checkpoint")
            return False

    def selfPlay(self, iteration=0, game_id=0):
        memory = []
        training_samples = []
        state = self.game.get_initial_state()
        move_count = 0
        play_history = []

        while True:
            action_probs = self.mcts.search(state, add_noise=True)
            if not isinstance(action_probs, np.ndarray):
                action_probs = np.array(action_probs, dtype=np.float32)

            # Temperatura decreciente con el avance de la partida
            if move_count < 30:
                temperature = 1.0
            elif move_count < 60:
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
               if is_terminal:
                if value == 0:
                    winner = 'draw'
                else:
    
                    winner = 'white' if (state.turn == False) else 'black'

                returnMemory = []
                
                for idx, (hist_state, hist_action_probs, hist_turn, hist_chosen_action) in enumerate(play_history):
                    
                    if winner == 'draw':
                        hist_outcome = 0.0
                        
                    elif winner == 'white':
                     
                        hist_outcome = 1.0 if hist_turn else -1.0
                        
                    else: 
                        hist_outcome = -1.0 if hist_turn else 1.0

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
                        elif isinstance(played_move, str):
                            move_uci = played_move
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

                try:
                    self.logger.log_training_data(iteration, game_id, training_samples, self.game) # type: ignore
                except Exception as e:
                    logging.error(f"Error guardando training data: {e}")

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
                try:
                    self.logger.log_game_stats(iteration, game_id, stats) # type: ignore
                except Exception as e:
                    logging.error(f"Error guardando stats de la partida: {e}")

                return returnMemory

    def train(self, memory):        
        # Crear el Dataset y el DataLoader
        dataset = ChessDataset(memory)

        dataloader = DataLoader(
            dataset, 
            batch_size=self.args['batch_size'], 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        self.model.train()

        # El bucle ahora es mucho más limpio y rápido
        for batch in dataloader:
            state_batch, policy_targets_batch, value_targets_batch = batch

            # Mover a dispositivo (XPU/GPU)
            # non_blocking=True permite que la transferencia sea asíncrona
            state = state_batch.to(self.device, non_blocking=True)
            policy_targets = policy_targets_batch.to(self.device, non_blocking=True)
            value_targets = value_targets_batch.to(self.device, non_blocking=True).unsqueeze(1)

            # Forward pass
            out_policy, out_value = self.model(state)
            
            # Loss calculation
            policy_loss = -torch.sum(policy_targets * F.log_softmax(out_policy, dim=1)) / policy_targets.size(0)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Backward pass
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
        # Configuración: guardar cada X iteraciones
        SAVE_EVERY = self.args.get('save_every', 5)  # Por defecto cada 5 iteraciones
        start_iter = self.args.get('start_iteration', 0)
        
        logging.info(f"\n{'='*70}")
        logging.info(f"CONFIGURACIÓN DE GUARDADO")
        logging.info(f"{'='*70}")
        logging.info(f"Se guardará checkpoint cada {SAVE_EVERY} iteraciones")
        logging.info(f"{'='*70}\n")
        
        for iteration in range(start_iter, self.args['num_iterations']):
            logging.info(f"\n{'='*60}")
            logging.info(f"ITERACIÓN {iteration + 1}/{self.args['num_iterations']}")
            logging.info('='*60)

            memory = []
            self.model.eval()

            logging.info(f"\nGenerando datos con self-play ({self.args['num_selfPlay_iterations']} partidas)")
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations']), desc="Self-play"):
                game_memory = self.selfPlay(iteration=iteration, game_id=selfPlay_iteration)
                memory += game_memory

            logging.info(f" Generados {len(memory)} estados de entrenamiento")

            # Mostrar resumen de partidas
            summary = self.logger.get_game_summary(iteration)
            if summary:
                logging.info(f"\nResumen de partidas:")
                logging.info(f"   Blancas: {summary.get('white_wins', 0)} | "
                      f"Negras: {summary.get('black_wins', 0)} | "
                      f"Empates: {summary.get('draws', 0)}")
                if 'avg_moves' in summary:
                    try:
                        logging.info(f"   Promedio de movimientos: {summary['avg_moves']:.1f}")
                    except Exception:
                        pass

        # Entrenamiento
        self.model.train()
        logging.info(f"\nEntrenando modelo ({self.args['num_epochs']} épocas)")

        # Variables para trackear el loss final
        final_policy_loss = 0.0
        final_value_loss = 0.0

        for epoch in range(self.args['num_epochs']):
            avg_policy_loss, avg_value_loss = self.train(memory)
            
            # Guardar los loss de la última época
            final_policy_loss = avg_policy_loss
            final_value_loss = avg_value_loss
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"   Época {epoch + 1}/{self.args['num_epochs']}: "
                            f"Policy Loss = {avg_policy_loss:.4f}, "
                            f"Value Loss = {avg_value_loss:.4f}")

        # ← AGREGAR ESTO (el logging que faltaba):
        logging.info(f"\n   Loss final de la iteración:")
        logging.info(f"   Policy Loss: {final_policy_loss:.4f}")
        logging.info(f"   Value Loss: {final_value_loss:.4f}")

        # Guardar checkpoint si corresponde
        should_save = (iteration + 1) % SAVE_EVERY == 0 or iteration == self.args['num_iterations'] - 1

        if should_save:
            success = self._save_checkpoint_bundle(iteration)
            
            if not success:
                logging.info(f"ADVERTENCIA: No se pudo guardar checkpoint de iteración {iteration}")
                logging.info(f"El entrenamiento continuará, pero se perdió este punto de guardado")

        logging.info(f"\n✓ Iteración {iteration + 1} completada")
        
        # Resumen final
        logging.info("\n" + "="*70)
        logging.info("ENTRENAMIENTO COMPLETADO")
        logging.info("="*70)
        self._print_final_summary()
    
    def _print_final_summary(self):
  
        logging.info("\nModelos guardados:")
        
        # Listar todos los archivos .pt en el directorio
        model_files = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt') and not f.endswith('_optimizer.pt')])
        
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
