import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Conexión residual
        x = F.relu(x)
        return x


class ChessResNet(nn.Module):
    
    def __init__(self, game, num_resBlocks, num_hidden, input_channels=12):
        super().__init__()
        
        self.game = game
        self.board_size = game.row_count  
        self.action_size = game.action_size  
    

        self.startBlock = nn.Sequential(
            nn.Conv2d(input_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        # POLICY HEAD -
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.board_size * self.board_size, self.action_size)
        )
        
        # VALUE HEAD 
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.board_size * self.board_size, 1),
            nn.Tanh()  # Comprime output a rango [-1, 1]
        )
        
    def forward(self, x):
     
        x = self.startBlock(x)
        
        for resBlock in self.backBone:
            x = resBlock(x)
        
        policy = self.policyHead(x)  
        value = self.valueHead(x)    
        
        return policy, value


def create_chess_model(game, num_resBlocks=9, num_hidden=128):
 
    model = ChessResNet(
        game=game,
        num_resBlocks=num_resBlocks,
        num_hidden=num_hidden,
        input_channels=12  # 6 tipos de piezas x 2 colores
    )
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

