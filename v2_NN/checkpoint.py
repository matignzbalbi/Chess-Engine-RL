import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import torch
import os
import json
from pathlib import Path
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero


def load_checkpoint(checkpoint_path, game, device='cpu'):
 
    
    # Verificar que existe el checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    
    logging.info(f"\n{'='*70}")
    logging.info(f"CARGANDO CHECKPOINT")
    logging.info(f"{'='*70}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    
    # Obtener rutas relacionadas
    base_path = checkpoint_path.replace('.pt', '')
    config_path = f"{base_path}_config.json"
    optimizer_path = f"{base_path}_optimizer.pt"
    
    # 1. Cargar configuración
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"✓ Configuración cargada desde JSON")
    else:
        # Configuración por defecto si no existe
        logging.warning(f"⚠️  No se encontró {config_path}, usando valores por defecto")
        config = {
            'num_resBlocks': 12,
            'num_hidden': 256,
            'iteration': 0
        }
    
    num_resBlocks = config.get('num_resBlocks', 12)
    num_hidden = config.get('num_hidden', 256)
    last_iteration = config.get('iteration', 0)
    
    logging.info(f"  • num_resBlocks: {num_resBlocks}")
    logging.info(f"  • num_hidden: {num_hidden}")
    logging.info(f"  • Última iteración completada: {last_iteration}")
    
    # 2. Crear modelo con la arquitectura correcta
    logging.info(f"\nCreando modelo...")
    model = create_chess_model(
        game=game,
        num_resBlocks=num_resBlocks,
        num_hidden=num_hidden
    )
    
    # 3. Cargar pesos del modelo
    logging.info(f"Cargando pesos del modelo...")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"✓ Modelo cargado exitosamente")
    except Exception as e:
        logging.error(f"❌ Error cargando modelo: {e}")
        raise
    
    # 4. Crear optimizer
    logging.info(f"\nCreando optimizer...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    
    # 5. Cargar estado del optimizer (si existe)
    if os.path.exists(optimizer_path):
        try:
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
            logging.info(f"✓ Estado del optimizer cargado")
        except Exception as e:
            logging.warning(f"⚠️  No se pudo cargar optimizer: {e}")
            logging.info(f"   Se continuará con optimizer reiniciado")
    else:
        logging.info(f"⚠️  No se encontró archivo de optimizer")
        logging.info(f"   Se continuará con optimizer desde cero")
    
    logging.info(f"{'='*70}\n")
    
    return model, optimizer, config


def continue_training_from_checkpoint(
    checkpoint_path,
    additional_iterations=10,
    num_selfPlay_iterations=20,
    num_searches=200,
    num_epochs=10,
    batch_size=512,
    save_every=5,
    C=4.0
):

    
    # Configurar device
    if torch.xpu.is_available():
        device = torch.device("xpu")
        logging.info("✓ Usando Intel GPU (XPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("✓ Usando NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        logging.info("✓ Usando CPU")
    
    # Inicializar juego
    INCLUDE_QUEEN_PROMOTIONS = False
    game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)
    
    # Cargar checkpoint
    model, optimizer, config = load_checkpoint(checkpoint_path, game, device=str(device))
    
    # Determinar desde qué iteración comenzar
    last_iteration = config.get('iteration', 0)
    start_iteration = last_iteration + 1
    total_iterations = start_iteration + additional_iterations
    
    logging.info(f"\n{'='*70}")
    logging.info(f"CONFIGURACIÓN DE CONTINUACIÓN")
    logging.info(f"{'='*70}")
    logging.info(f"Última iteración completada: {last_iteration}")
    logging.info(f"Comenzando desde iteración: {start_iteration}")
    logging.info(f"Iteraciones adicionales: {additional_iterations}")
    logging.info(f"Total de iteraciones al finalizar: {total_iterations - 1}")
    logging.info(f"{'='*70}\n")
    
    # Configurar argumentos para AlphaZero
    args = {
        'C': C,
        'num_searches': num_searches,
        'num_selfPlay_iterations': num_selfPlay_iterations,
        'num_iterations': total_iterations,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'save_every': save_every,
        'start_iteration': start_iteration  # ← Importante: comenzar desde aquí
    }
    
    logging.info(f"Parámetros de entrenamiento:")
    logging.info(f"  • C (exploración): {C}")
    logging.info(f"  • Búsquedas MCTS: {num_searches}")
    logging.info(f"  • Partidas por iteración: {num_selfPlay_iterations}")
    logging.info(f"  • Épocas por iteración: {num_epochs}")
    logging.info(f"  • Batch size: {batch_size}")
    logging.info(f"  • Guardar cada: {save_every} iteraciones\n")
    
    # Crear instancia de AlphaZero
    alphaZero = AlphaZero(model, optimizer, game, args)
    
    # Iniciar entrenamiento
    logging.info(f"{'='*70}")
    logging.info(f"INICIANDO ENTRENAMIENTO CONTINUO")
    logging.info(f"{'='*70}\n")
    
    try:
        alphaZero.learn()
        logging.info("\n✓ Entrenamiento completado exitosamente")
    except KeyboardInterrupt:
        logging.info("\n⚠️  Entrenamiento interrumpido por el usuario")
    except Exception as e:
        logging.error(f"\n❌ Error durante el entrenamiento: {e}")
        raise
    
    return model


if __name__ == "__main__":
    
    # 1. Especificar el checkpoint desde el que querés continuar
    CHECKPOINT_PATH = "/home/mgbalbi/UNLu-MCTSNN/jobs/pytorch_files_42705/model_34.pt" 
    
    # 2. Configurar parámetros de continuación
    continue_training_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        additional_iterations=200,      # 20 iteraciones más
        num_selfPlay_iterations=25,    # 20 partidas por iteración
        num_searches=250,              # 200 búsquedas MCTS
        num_epochs=10,                 # 10 épocas
        batch_size=512,                # Batch size
        save_every=2,                  # Guardar cada 5 iteraciones
        C=4.0                          # Exploración MCTS
    )
    
    logging.info("\n✓ Script finalizado")