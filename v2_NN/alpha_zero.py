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

        # MCTS
        self.mcts = MCTS(game, args, model, device=self.device)

        # Logger
        self.logger = GameLogger()
        print(f"Logs guardándose en: {self.logger.log_dir}/")
        
        # ✨ NUEVO: Verificar directorio de checkpoints al inicio
        self.checkpoint_dir = "pytorch_files"
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
            
            print(f"Directorio de checkpoints verificado: {self.checkpoint_dir}/")
            
        except PermissionError:
            print(f"ERROR: Sin permisos de escritura en {self.checkpoint_dir}/")
            raise
        except OSError as e:
            print(f"ERROR: No se puede crear directorio {self.checkpoint_dir}/: {e}")
            raise
    
    def _get_available_disk_space(self, path="."):
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
     
        # Paso 1: Verificar espacio en disco
        estimated_size = self._estimate_checkpoint_size()
        available_space = self._get_available_disk_space(self.checkpoint_dir)
        
        if available_space < estimated_size * 2:  # Requerir 2x el tamaño (seguridad)
            print(f"⚠️  ADVERTENCIA: Poco espacio en disco")
            print(f"   Disponible: {available_space:.1f} MB")
            print(f"   Necesario: {estimated_size * 2:.1f} MB")
            print(f"   NO se guardó {description}")
            return False
        
        try:
            # Paso 2: Crear backup si el archivo ya existe
            backup_path = None
            if os.path.exists(filepath):
                backup_path = filepath + ".backup"
                try:
                    shutil.copy2(filepath, backup_path)
                except Exception as e:
                    print(f"No se pudo crear backup de {filepath}: {e}")
                    # Continuar de todas formas
            
            # Paso 3: Guardar en archivo temporal (guardado atómico)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.pt',
                dir=self.checkpoint_dir,
                prefix='.tmp_'
            )
            os.close(temp_fd)  # Cerrar el file descriptor
            
            # Guardar en temporal
            torch.save(obj, temp_path)
            
            # Paso 4: Verificar que el archivo temporal se escribió correctamente
            if not os.path.exists(temp_path):
                raise IOError(f"Archivo temporal {temp_path} no se creó")
            
            temp_size = os.path.getsize(temp_path)
            if temp_size < 1000:  # Menos de 1KB es sospechoso
                raise IOError(f"Archivo temporal muy pequeño: {temp_size} bytes")
            
            # Paso 5: Mover archivo temporal al destino final (operación atómica)
            shutil.move(temp_path, filepath)
            
            # Paso 6: Verificar que el archivo final existe
            if not os.path.exists(filepath):
                raise IOError(f"Archivo final {filepath} no existe después de mover")
            
            # Paso 7: Eliminar backup si todo salió bien
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass  # No crítico si falla
            
            return True
            
        except IOError as e:
            print(f"ERROR de E/S al guardar {description}: {e}")
            
            # Restaurar desde backup si existe
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, filepath)
                    print(f"✅ Restaurado desde backup: {filepath}")
                except Exception as restore_err:
                    print(f"No se pudo restaurar backup: {restore_err}")
            
            return False
            
        except RuntimeError as e:
            print(f"ERROR de PyTorch al guardar {description}: {e}")
            return False
            
        except MemoryError:
            print(f"ERROR: Sin memoria para guardar {description}")
            return False
            
        except Exception as e:
            print(f"ERROR inesperado al guardar {description}: {type(e).__name__}: {e}")
            return False
        
        finally:
            # Limpiar archivos temporales si quedaron
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
    
    def _save_checkpoint_bundle(self, iteration, difficulty_level=None):
  
        success_count = 0
        total_saves = 3  # modelo + optimizer + config
        
        # Determinar nombres de archivos
        if difficulty_level:
            base_name = f"bot_{difficulty_level}"
            print(f"\n🎮 Guardando checkpoint de nivel: {difficulty_level.upper()}")
        else:
            base_name = f"model_{iteration}"
            print(f"\nGuardando checkpoint de iteración {iteration}")
        
        model_path = os.path.join(self.checkpoint_dir, f"{base_name}.pt")
        optimizer_path = os.path.join(self.checkpoint_dir, f"{base_name}_optimizer.pt")
        config_path = os.path.join(self.checkpoint_dir, f"{base_name}_config.json")
        
        # Guardar modelo
        print(f"Guardando modelo")
        if self._safe_save_checkpoint(self.model.state_dict(), model_path, f"modelo {base_name}"):
            size_mb = os.path.getsize(model_path) / (1024 ** 2)
            print(f"   ✅ Modelo guardado ({size_mb:.1f} MB)")
            success_count += 1
        else:
            print(f"Falló guardado de modelo")
        
        # Guardar optimizer
        print(f"Guardando optimizer")
        if self._safe_save_checkpoint(self.optimizer.state_dict(), optimizer_path, f"optimizer {base_name}"):
            size_mb = os.path.getsize(optimizer_path) / (1024 ** 2)
            print(f"   ✅ Optimizer guardado ({size_mb:.1f} MB)")
            success_count += 1
        else:
            print(f"Falló guardado de optimizer")
        
        # Guardar configuración (JSON)
        print(f"Guardando configuración")
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
            
            if difficulty_level:
                config['level'] = difficulty_level
                config['recommended_elo'] = {
                    'principiante': '600-800',
                    'intermedio': '1000-1200',
                    'avanzado': '1400-1600'
                }.get(difficulty_level, 'unknown')
            
            # Guardar JSON con guardado atómico también
            temp_config = config_path + ".tmp"
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            shutil.move(temp_config, config_path)
            print(f"Configuración guardada")
            success_count += 1
            
        except Exception as e:
            print(f"Falló guardado de configuración: {e}")
        
        # Resultado final
        if success_count == total_saves:
            print(f"Checkpoint completo guardado exitosamente")
            if difficulty_level:
                print(f"Bot {difficulty_level} listo para usar")
            return True
        elif success_count > 0:
            print(f"Checkpoint guardado parcialmente ({success_count}/{total_saves})")
            return False
        else:
            print(f"Falló completamente el guardado del checkpoint")
            return False

    # ============================================================================
    # MÉTODOS ORIGINALES (selfPlay y train sin cambios)
    # ============================================================================

    def selfPlay(self, iteration=0, game_id=0):
        """[Código original sin cambios]"""
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
                    print(f"Error guardando training data: {e}")

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
                    print(f"Error guardando stats de la partida: {e}")

                return returnMemory

    def train(self, memory):
        """[Código original sin cambios]"""
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

    # ============================================================================
    # LEARN CON GUARDADO SEGURO
    # ============================================================================

    def learn(self):
        """Loop principal de aprendizaje con guardado robusto"""
        
        # Configuración de guardado
        SAVE_EVERY = 5
        DIFFICULTY_CHECKPOINTS = {
            'principiante': 9,
            'intermedio': 29,
            'avanzado': 49
        }
        
        start_iter = self.args.get('start_iteration', 0)
        
        for iteration in range(start_iter, self.args['num_iterations']):
            print(f"\n{'='*60}")
            print(f"ITERACIÓN {iteration + 1}/{self.args['num_iterations']}")
            print('='*60)

            memory = []
            self.model.eval()

            print(f"\nGenerando datos con self-play ({self.args['num_selfPlay_iterations']} partidas)")
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations']), desc="Self-play"):
                game_memory = self.selfPlay(iteration=iteration, game_id=selfPlay_iteration)
                memory += game_memory

            print(f"Generados {len(memory)} estados de entrenamiento")

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
                        pass

            self.model.train()
            print(f"\nEntrenando modelo ({self.args['num_epochs']} épocas)")

            for epoch in range(self.args['num_epochs']):
                avg_policy_loss, avg_value_loss = self.train(memory)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Época {epoch + 1}/{self.args['num_epochs']}: "
                          f"Policy Loss = {avg_policy_loss:.4f}, "
                          f"Value Loss = {avg_value_loss:.4f}")

            should_save = False
            difficulty_level = None
            
            for level, checkpoint_iter in DIFFICULTY_CHECKPOINTS.items():
                if iteration == checkpoint_iter:
                    should_save = True
                    difficulty_level = level
                    break
            
            if not should_save:
                if (iteration + 1) % SAVE_EVERY == 0 or iteration == self.args['num_iterations'] - 1:
                    should_save = True
            
            if should_save:
                success = self._save_checkpoint_bundle(iteration, difficulty_level)
                
                # Si es checkpoint de dificultad y falló, intentar guardar checkpoint regular
                if difficulty_level and not success:
                    print(f"Intentando guardar checkpoint regular en su lugar...")
                    self._save_checkpoint_bundle(iteration, difficulty_level=None)
                
                # Si falló completamente, advertir pero continuar entrenamiento
                if not success:
                    print(f"ADVERTENCIA: No se pudo guardar checkpoint de iteración {iteration}")
                    print(f"El entrenamiento continuará, pero se perdió este punto de guardado")
            
            print(f"Iteración {iteration + 1} completada")
        
        # Resumen final
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
        self._print_final_summary(DIFFICULTY_CHECKPOINTS)
    
    def _print_final_summary(self, difficulty_checkpoints):
        print("\nModelos de dificultad generados:")
        
        for level, checkpoint_iter in difficulty_checkpoints.items():
            path = os.path.join(self.checkpoint_dir, f"bot_{level}.pt")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024**2)
                print(f"{level.capitalize():15} → {path:40} ({size_mb:.1f} MB)")
            else:
                print(f"{level.capitalize():15} → NO GUARDADO")
        
        print("\n💡 Para usar estos bots:")
        print('   python pygame_interface.py --model pytorch_files/bot_principiante.pt')
        print("="*70)