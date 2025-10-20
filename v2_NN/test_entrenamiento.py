import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

game = ChessGame()
model = create_chess_model(game, num_resBlocks=2, num_hidden=32) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

args = {
    'C': 2,
    'num_searches': 20,
    'num_iterations': 5,
    'num_selfPlay_iterations': 1,
    'num_epochs': 1,
    'batch_size': 16
}

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()
