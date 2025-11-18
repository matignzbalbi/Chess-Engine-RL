import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero
from ddp_utils import setup_distributed, wrap_model_ddp, cleanup_distributed

INCLUDE_QUEEN_PROMOTIONS = False


rank, world_size, local_rank = setup_distributed(backend='ccl')
device = torch.device(f'xpu:{local_rank}') if local_rank is not None else torch.device('xpu:0')
game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)
model = create_chess_model(game=game, num_resBlocks=4, num_hidden=64)
model = wrap_model_ddp(model, device, device_type='xpu', local_rank=local_rank)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'C': 2,
    'num_searches': 100,
    'num_selfPlay_iterations': 10,
    'num_iterations': 3,
    'num_epochs': 2,
    'batch_size': 32,
    'save_every': 1
}

logging.info(f"Action size: {game.action_size}")
logging.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
logging.info(f"Config: {args['num_iterations']} iterations, {args['num_searches']} searches/move\n")

alphaZero = AlphaZero(model, optimizer, game, args, rank=rank, world_size=world_size) # type: ignore
alphaZero.learn()

logging.info("Finalizado")

cleanup_distributed()
