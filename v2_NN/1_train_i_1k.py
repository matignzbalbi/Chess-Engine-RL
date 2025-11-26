import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

# Resumén de los cambios  

# Prueba pesada con 1000 búsquedas para intentar mitigar las tablas constantes.

INCLUDE_QUEEN_PROMOTIONS = False

game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)
model = create_chess_model(game=game, num_resBlocks=12, num_hidden=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'C': 3,
    'num_searches': 1000,
    'num_selfPlay_iterations': 5,
    'num_iterations': 100,
    'num_epochs': 15,
    'batch_size': 512,
    'save_every': 1
}

logging.info(f"Action size: {game.action_size}")
logging.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
logging.info(f"Config: {args['num_iterations']} iterations, {args['num_searches']} searches/move\n")

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()

logging.info("Finalizado")