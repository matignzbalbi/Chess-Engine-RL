import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

game = ChessGame()
model = create_chess_model(game, num_resBlocks=4, num_hidden=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

args = {
    'C': 2,
    'num_searches': 100,
    'num_iterations': 10,
    'num_selfPlay_iterations': 20,
    'num_epochs': 4,
    'batch_size': 32
}

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()
