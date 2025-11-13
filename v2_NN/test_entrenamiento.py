import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

# Configuración de promociones
INCLUDE_QUEEN_PROMOTIONS = False

# Setup
game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)
model = create_chess_model(game=game, num_resBlocks=4, num_hidden=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'C': 2,
    'num_searches': 50,
    'num_selfPlay_iterations': 20,
    'num_iterations': 50,
    'num_epochs': 4,
    'batch_size': 64
}

print(f"Action size: {game.action_size}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Config: {args['num_iterations']} iterations, {args['num_searches']} searches/move\n")

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()

print("Finalizado")