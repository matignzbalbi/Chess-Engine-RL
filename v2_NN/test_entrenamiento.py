import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

# Configuración de promociones
INCLUDE_QUEEN_PROMOTIONS = False

# Setup
game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)
model = create_chess_model(game=game, num_resBlocks=2, num_hidden=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'C': 2,
    'num_searches': 5,
    'num_selfPlay_iterations': 1,
    'num_iterations': 5,
    'num_epochs': 1,
    'batch_size': 16
}

logging.info(f"Action size: {game.action_size}")
logging.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
logging.info(f"Config: {args['num_iterations']} iterations, {args['num_searches']} searches/move\n")

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()

logging.info("Finalizado")