import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Bloque residual básico"""
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
    """
    Red neuronal ResNet para ajedrez, similar a AlphaZero.
    
    Arquitectura:
    - Input: 8x8x12 (tablero con 12 canales: 6 tipos de piezas x 2 colores)
    - Bloque inicial: Conv + BatchNorm + ReLU
    - N bloques residuales
    - Dos cabezas:
        1. Policy head: predice probabilidades de movimientos
        2. Value head: predice quién va ganando (-1 a +1)
    """
    
    def __init__(self, game, num_resBlocks, num_hidden, input_channels=12):
        super().__init__()
        
        self.game = game
        self.board_size = game.row_count  # 8 para ajedrez
        self.action_size = game.action_size  # 4672 para ajedrez
        
        # BLOQUE INICIAL
        # Convierte el input de 12 canales a num_hidden canales
        self.startBlock = nn.Sequential(
            nn.Conv2d(input_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # BACKBONE - Torre de bloques residuales
        # Similar a ResNet: permite que la red aprenda características complejas
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        # POLICY HEAD - Predice qué movimiento hacer
        # Output: probabilidad para cada uno de los 4672 movimientos posibles
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.board_size * self.board_size, self.action_size)
        )
        
        # VALUE HEAD - Predice quién va ganando
        # Output: un valor entre -1 (negras ganan) y +1 (blancas ganan)
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.board_size * self.board_size, 1),
            nn.Tanh()  # Comprime output a rango [-1, 1]
        )
        
    def forward(self, x):
        """
        Forward pass de la red.
        
        Args:
            x: Tensor de forma (batch_size, 12, 8, 8)
               Representa el tablero codificado
        
        Returns:
            policy: Tensor de forma (batch_size, 4672)
                   Logits para cada movimiento posible
            value: Tensor de forma (batch_size, 1)
                  Valor de la posición entre -1 y 1
        """
        # 1. Bloque inicial
        x = self.startBlock(x)
        
        # 2. Bloques residuales
        for resBlock in self.backBone:
            x = resBlock(x)
        
        # 3. Dos cabezas separadas
        policy = self.policyHead(x)  # Probabilidades de movimientos
        value = self.valueHead(x)    # Evaluación de posición
        
        return policy, value


def create_chess_model(game, num_resBlocks=9, num_hidden=128):
    """
    Factory function para crear el modelo.
    
    Args:
        game: Instancia de ChessGame
        num_resBlocks: Número de bloques residuales (AlphaZero usa 19-40)
        num_hidden: Canales en capas ocultas (AlphaZero usa 256)
    
    Returns:
        Modelo de PyTorch
    """
    model = ChessResNet(
        game=game,
        num_resBlocks=num_resBlocks,
        num_hidden=num_hidden,
        input_channels=12  # 6 tipos de piezas x 2 colores
    )
    return model


# Función auxiliar para contar parámetros
def count_parameters(model):
    """Cuenta el número de parámetros entrenables del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Prueba simple del modelo
    from chess_game import ChessGame
    
    game = ChessGame()
    model = create_chess_model(game, num_resBlocks=4, num_hidden=64)
    
    print("=== ARQUITECTURA DEL MODELO ===")
    print(model)
    print(f"\nNúmero de parámetros: {count_parameters(model):,}")
    
    # Probar forward pass
    print("\n=== PRUEBA DE FORWARD PASS ===")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 12, 8, 8)
    
    policy, value = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Policy output shape: {policy.shape}")  # (2, 4672)
    print(f"Value output shape: {value.shape}")    # (2, 1)
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")