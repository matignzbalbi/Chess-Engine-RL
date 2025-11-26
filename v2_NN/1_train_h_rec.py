import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

# Resumén de los cambios  

# Aumentamos ligeramente C y los párametros de búsqueda para comparar los tiempos de entrenamiento.
# En este entrenamiento realizamos cambios en el sistema de recompensas penalizando las tablas ligeramente.

INCLUDE_QUEEN_PROMOTIONS = False

game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)
model = create_chess_model(game=game, num_resBlocks=12, num_hidden=256) #
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 

args = {
    'C': 5, # Subimos
    'num_searches': 350, # Subimos
    'num_selfPlay_iterations': 20, 
    'num_iterations': 100, 
    'num_epochs': 10, 
    'batch_size': 512,
    'save_every': 2
}

logging.info(f"Action size: {game.action_size}")
logging.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
logging.info(f"Config: {args['num_iterations']} iterations, {args['num_searches']} searches/move\n")

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()

logging.info("Finalizado")