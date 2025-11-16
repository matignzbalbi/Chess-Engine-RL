import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import numpy as np
import os
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from chess_game import ChessGame
from model import create_chess_model
import chess


class DeadNeuronAnalyzer:
    """
    Analizador de neuronas muertas en modelos de ajedrez.
    Identifica neuronas que nunca se activan (ReLU = 0 siempre).
    """
    
    def __init__(self, model, game, device='cpu'):
        self.model = model
        self.game = game
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Diccionario para almacenar activaciones por capa
        self.activations = defaultdict(list)
        self.hooks = []
        
    def register_hooks(self):
        """Registra hooks en todas las capas ReLU y BatchNorm"""
        
        def hook_fn(name):
            def hook(module, input, output):
                # Solo capturar la salida (después de ReLU)
                if isinstance(output, torch.Tensor):
                    # Flatten y guardar como numpy
                    flat = output.detach().cpu().flatten().numpy()
                    self.activations[name].append(flat)
            return hook
        
        # Registrar hooks en todas las capas
        for name, module in self.model.named_modules():
            # Solo en ReLU y BatchNorm (donde se puede ver activación)
            if isinstance(module, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                
        logging.info(f"✓ Hooks registrados en {len(self.hooks)} capas")
    
    def remove_hooks(self):
        """Remueve todos los hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logging.info("✓ Hooks removidos")
    
    def analyze_positions(self, num_positions=100, use_random=True, use_real_games=True):
        """
        Analiza múltiples posiciones de ajedrez
        
        Args:
            num_positions: Cantidad de posiciones a analizar
            use_random: Si incluir movimientos aleatorios
            use_real_games: Si incluir posiciones de partidas simuladas
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"ANALIZANDO {num_positions} POSICIONES")
        logging.info(f"{'='*60}\n")
        
        positions_analyzed = 0
        
        with torch.no_grad():
            # 1. Posición inicial
            state = self.game.get_initial_state()
            self._process_position(state)
            positions_analyzed += 1
            
            # 2. Posiciones aleatorias
            if use_random:
                logging.info("Generando posiciones aleatorias...")
                for i in tqdm(range(num_positions // 2), desc="Random"):
                    state = self._generate_random_position()
                    if state:
                        self._process_position(state)
                        positions_analyzed += 1
            
            # 3. Posiciones de partidas reales
            if use_real_games:
                logging.info("\nGenerando posiciones de partidas simuladas...")
                remaining = num_positions - positions_analyzed
                for i in tqdm(range(remaining), desc="Games"):
                    state = self._generate_game_position()
                    if state:
                        self._process_position(state)
                        positions_analyzed += 1
        
        logging.info(f"\n✓ {positions_analyzed} posiciones analizadas")
    
    def _process_position(self, state):
        """Procesa una posición: ejecuta forward pass"""
        try:
            encoded = self.game.get_encoded_state(state)
            tensor = torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
            _ = self.model(tensor)
        except Exception as e:
            logging.warning(f"Error procesando posición: {e}")
    
    def _generate_random_position(self, max_moves=30):
        """Genera una posición aleatoria con movimientos legales"""
        state = self.game.get_initial_state()
        num_moves = np.random.randint(1, max_moves)
        
        for _ in range(num_moves):
            legal_moves = list(state.legal_moves)
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
            state.push(move)
            
            # Si termina la partida, devolver
            if state.is_game_over():
                break
        
        return state
    
    def _generate_game_position(self, max_moves=60):
        """Genera posición jugando una mini-partida aleatoria"""
        state = self.game.get_initial_state()
        num_moves = np.random.randint(5, max_moves)
        
        for _ in range(num_moves):
            legal_moves = list(state.legal_moves)
            if not legal_moves or state.is_game_over():
                break
            
            # Selección sesgada: mayor probabilidad a capturas y jaques
            captures = [m for m in legal_moves if state.is_capture(m)]
            checks = [m for m in legal_moves if state.gives_check(m)]
            
            if captures and np.random.random() < 0.3:
                move = np.random.choice(captures)
            elif checks and np.random.random() < 0.2:
                move = np.random.choice(checks)
            else:
                move = np.random.choice(legal_moves)
            
            state.push(move)
        
        return state
    
    def compute_statistics(self):
        """Calcula estadísticas de neuronas muertas"""
        
        logging.info(f"\n{'='*60}")
        logging.info("CALCULANDO ESTADÍSTICAS")
        logging.info(f"{'='*60}\n")
        
        results = {}
        
        for layer_name, activations_list in self.activations.items():
            if not activations_list:
                continue
            
            # Concatenar todas las activaciones de esta capa
            all_activations = np.concatenate(activations_list, axis=0)
            
            # Reshape para tener (num_samples, num_neurons)
            num_samples = len(activations_list)
            num_neurons = len(activations_list[0])
            activations_matrix = np.array(activations_list)  # (num_samples, num_neurons)
            
            # Calcular métricas
            mean_activation = activations_matrix.mean(axis=0)
            max_activation = activations_matrix.max(axis=0)
            std_activation = activations_matrix.std(axis=0)
            
            # Neuronas muertas: nunca activadas (max = 0)
            dead_neurons = (max_activation == 0).sum()
            dead_percentage = (dead_neurons / num_neurons) * 100
            
            # Neuronas casi muertas: activación promedio < 0.01
            nearly_dead = (mean_activation < 0.01).sum()
            nearly_dead_percentage = (nearly_dead / num_neurons) * 100
            
            # Sparsity: porcentaje de activaciones = 0
            sparsity = (activations_matrix == 0).sum() / activations_matrix.size * 100
            
            results[layer_name] = {
                'total_neurons': num_neurons,
                'dead_neurons': int(dead_neurons),
                'dead_percentage': float(dead_percentage),
                'nearly_dead': int(nearly_dead),
                'nearly_dead_percentage': float(nearly_dead_percentage),
                'sparsity': float(sparsity),
                'mean_activation': float(mean_activation.mean()),
                'std_activation': float(std_activation.mean()),
                'max_activation': float(max_activation.max()),
            }
        
        return results
    
    def print_summary(self, results):
        """Imprime resumen de resultados"""
        
        logging.info(f"\n{'='*60}")
        logging.info("RESUMEN DE NEURONAS MUERTAS")
        logging.info(f"{'='*60}\n")
        
        total_neurons = 0
        total_dead = 0
        total_nearly_dead = 0
        
        # Ordenar por porcentaje de muertas (descendente)
        sorted_layers = sorted(results.items(), 
                             key=lambda x: x[1]['dead_percentage'], 
                             reverse=True)
        
        for layer_name, stats in sorted_layers:
            total_neurons += stats['total_neurons']
            total_dead += stats['dead_neurons']
            total_nearly_dead += stats['nearly_dead']
            
            # Determinar severidad
            severity = self._get_severity(stats['dead_percentage'])
            
            logging.info(f"{layer_name}")
            logging.info(f"  Total neuronas: {stats['total_neurons']}")
            logging.info(f"  {severity} Muertas: {stats['dead_neurons']} ({stats['dead_percentage']:.1f}%)")
            logging.info(f"  Casi muertas: {stats['nearly_dead']} ({stats['nearly_dead_percentage']:.1f}%)")
            logging.info(f"  Sparsity: {stats['sparsity']:.1f}%")
            logging.info(f"  Activación media: {stats['mean_activation']:.4f}")
            logging.info(f"  Activación máxima: {stats['max_activation']:.4f}")
            logging.info("")
        
        # Resumen global
        logging.info(f"{'='*60}")
        logging.info("RESUMEN GLOBAL")
        logging.info(f"{'='*60}")
        logging.info(f"Total de neuronas analizadas: {total_neurons}")
        logging.info(f"Neuronas muertas: {total_dead} ({total_dead/total_neurons*100:.1f}%)")
        logging.info(f"Neuronas casi muertas: {total_nearly_dead} ({total_nearly_dead/total_neurons*100:.1f}%)")
        
        overall_health = self._get_overall_health(total_dead/total_neurons*100)
        logging.info(f"\n{overall_health}")
        logging.info(f"{'='*60}\n")
    
    def _get_severity(self, percentage):
        """Retorna emoji según severidad"""
        if percentage == 0:
            return "✓"
        elif percentage < 10:
            return "⚠️"
        elif percentage < 30:
            return "⚠️⚠️"
        else:
            return "❌"
    
    def _get_overall_health(self, dead_percentage):
        """Retorna diagnóstico general"""
        if dead_percentage < 5:
            return "✓ RED SALUDABLE: Muy pocas neuronas muertas"
        elif dead_percentage < 15:
            return "⚠️ RED ACEPTABLE: Porcentaje moderado de neuronas muertas"
        elif dead_percentage < 30:
            return "⚠️⚠️ RED PROBLEMÁTICA: Alto porcentaje de neuronas muertas"
        else:
            return "❌ RED CRÍTICA: Demasiadas neuronas muertas, considerar cambios de arquitectura"
    
    def plot_results(self, results, output_dir='dead_neurons_analysis'):
        """Genera gráficos de resultados"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Gráfico de barras: % de neuronas muertas por capa
        layer_names = list(results.keys())
        dead_percentages = [results[l]['dead_percentage'] for l in layer_names]
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(layer_names)), dead_percentages)
        
        # Colorear según severidad
        colors = ['green' if p < 10 else 'orange' if p < 30 else 'red' 
                 for p in dead_percentages]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Capa')
        plt.ylabel('% Neuronas Muertas')
        plt.title('Porcentaje de Neuronas Muertas por Capa')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Umbral crítico (20%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dead_neurons_by_layer.png", dpi=150)
        plt.close()
        
        # 2. Gráfico de sparsity
        sparsity = [results[l]['sparsity'] for l in layer_names]
        
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(layer_names)), sparsity, color='steelblue')
        plt.xlabel('Capa')
        plt.ylabel('Sparsity (%)')
        plt.title('Sparsity (% de activaciones = 0) por Capa')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sparsity_by_layer.png", dpi=150)
        plt.close()
        
        # 3. Comparación: muertas vs casi-muertas
        nearly_dead_percentages = [results[l]['nearly_dead_percentage'] for l in layer_names]
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        plt.figure(figsize=(14, 6))
        plt.bar(x - width/2, dead_percentages, width, label='Completamente muertas', color='red', alpha=0.7)
        plt.bar(x + width/2, nearly_dead_percentages, width, label='Casi muertas', color='orange', alpha=0.7)
        plt.xlabel('Capa')
        plt.ylabel('Porcentaje')
        plt.title('Neuronas Muertas vs Casi Muertas')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dead_vs_nearly_dead.png", dpi=150)
        plt.close()
        
        logging.info(f"✓ Gráficos guardados en: {output_dir}/")
    
    def save_results(self, results, output_file='dead_neurons_results.json'):
        """Guarda resultados en JSON"""
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"✓ Resultados guardados en: {output_file}")


def load_model_from_checkpoint(checkpoint_path, game):
    """Carga modelo desde checkpoint con su configuración"""
    
    # Intentar cargar config JSON
    config_path = checkpoint_path.replace('.pt', '_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        num_resBlocks = config.get('num_resBlocks', 2)
        num_hidden = config.get('num_hidden', 32)
        logging.info(f"✓ Configuración cargada desde JSON")
    else:
        # Defaults si no hay config
        num_resBlocks = 2
        num_hidden = 32
        logging.warning(f"⚠️ No se encontró config JSON, usando defaults")
    
    logging.info(f"  num_resBlocks: {num_resBlocks}")
    logging.info(f"  num_hidden: {num_hidden}")
    
    # Crear y cargar modelo
    model = create_chess_model(game, num_resBlocks=num_resBlocks, num_hidden=num_hidden)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logging.info(f"✓ Modelo cargado exitosamente")
    except Exception as e:
        logging.error(f"❌ Error cargando modelo: {e}")
        raise
    
    return model


def main():
    """Función principal"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar neuronas muertas en modelo de ajedrez')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Ruta al checkpoint (.pt)')
    parser.add_argument('--positions', type=int, default=100,
                       help='Número de posiciones a analizar (default: 100)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu o cuda (default: cpu)')
    parser.add_argument('--output', type=str, default='dead_neurons_analysis',
                       help='Directorio de salida (default: dead_neurons_analysis)')
    
    args = parser.parse_args()
    
    # Verificar checkpoint existe
    if not os.path.exists(args.checkpoint):
        logging.error(f"❌ Checkpoint no encontrado: {args.checkpoint}")
        return
    
    logging.info(f"\n{'='*60}")
    logging.info("ANALIZADOR DE NEURONAS MUERTAS")
    logging.info(f"{'='*60}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Posiciones: {args.positions}")
    logging.info(f"Device: {args.device}")
    logging.info(f"{'='*60}\n")
    
    # Cargar game y modelo
    logging.info("Cargando modelo...")
    game = ChessGame()
    model = load_model_from_checkpoint(args.checkpoint, game)
    
    # Crear analizador
    analyzer = DeadNeuronAnalyzer(model, game, device=args.device)
    
    # Registrar hooks
    analyzer.register_hooks()
    
    # Analizar posiciones
    analyzer.analyze_positions(num_positions=args.positions)
    
    # Remover hooks
    analyzer.remove_hooks()
    
    # Calcular estadísticas
    results = analyzer.compute_statistics()
    
    # Imprimir resumen
    analyzer.print_summary(results)
    
    # Generar gráficos
    analyzer.plot_results(results, output_dir=args.output)
    
    # Guardar resultados
    output_json = os.path.join(args.output, 'results.json')
    analyzer.save_results(results, output_file=output_json)
    
    logging.info(f"\n✓ Análisis completo!")


if __name__ == "__main__":
    main()