import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

INCLUDE_QUEEN_PROMOTIONS = False

game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)
model = create_chess_model(game=game, num_resBlocks=8, num_hidden=192) # Achicamos la red
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4) # Aumentamos regularización

args = {
    'C': 2,
    'num_searches': 250, # Aumentamos las busquedas 
    'num_selfPlay_iterations': 45,
    'num_iterations': 35,
    'num_epochs': 7,
    'batch_size': 256, # Batches más grandes
    'save_every': 1
}

logging.info(f"Action size: {game.action_size}")
logging.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
logging.info(f"Config: {args['num_iterations']} iterations, {args['num_searches']} searches/move\n")

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()

logging.info("Finalizado")