import torch
from chess_game import ChessGame
from model import create_chess_model
from alpha_zero import AlphaZero

# ===========================
# CONFIGURACIÓN DE PROMOCIONES
# ===========================
# Cambiar a True para usar mapeo extendido con promociones a dama explícitas
INCLUDE_QUEEN_PROMOTIONS = False

print("=" * 70)
print("CONFIGURACIÓN DE ENTRENAMIENTO")
print("=" * 70)
print(f"Promociones explícitas a dama: {INCLUDE_QUEEN_PROMOTIONS}")
print("=" * 70)

# Inicializar juego con configuración elegida
game = ChessGame(include_queen_promotions=INCLUDE_QUEEN_PROMOTIONS)

print(f"\n✓ Juego inicializado")
print(f"  Action size: {game.action_size}")

# Crear modelo con el action_size correcto
model = create_chess_model(
    game=game, 
    num_resBlocks=2,  # Modelo pequeño para pruebas
    num_hidden=32
)

print(f"\n✓ Modelo creado")
print(f"  Bloques residuales: 2")
print(f"  Canales ocultos: 32")
print(f"  Parámetros: {sum(p.numel() for p in model.parameters()):,}")

# Optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Argumentos de entrenamiento
args = {
    'C': 2,                          # Constante de exploración UCB
    'num_searches': 50,              # Búsquedas MCTS por movimiento
    'num_iterations': 5,             # Iteraciones de entrenamiento
    'num_selfPlay_iterations': 1,    # Partidas por iteración
    'num_epochs': 1,                 # Épocas de entrenamiento por iteración
    'batch_size': 16                 # Tamaño de batch
}

print(f"\n✓ Configuración de entrenamiento:")
for key, value in args.items():
    print(f"  {key}: {value}")

# Inicializar AlphaZero
alphaZero = AlphaZero(model, optimizer, game, args)

print(f"\n{'='*70}")
print("INICIANDO ENTRENAMIENTO")
print('='*70)

# Entrenar
alphaZero.learn()

print(f"\n{'='*70}")
print("✅ ENTRENAMIENTO COMPLETADO")
print('='*70)