# Chess Engine - Implementación con Aprendizaje por Refuerzo

Motor de ajedrez basado en aprendizaje por refuerzo profundo siguiendo el algoritmo AlphaZero, entrenado en **Clementina XXI** (clúster de GPUs Intel Arc A770) utilizando Intel Extension for PyTorch (IPEX).

## 🔧 Stack Tecnológico
- **PyTorch** + **Intel Extension for PyTorch (IPEX)** para aceleración XPU
- **Monte Carlo Tree Search (MCTS)** guiado por red neuronal
- **Arquitectura ResNet** (12-20 bloques, 256 canales ocultos)
- Pipeline de entrenamiento mediante autopartidas con logging y sistema de rating ELO
- Interfaz Pygame para partidas humano vs IA

## 🎯 Características Principales
- Mapeo personalizado de movimientos para ajedrez (4672 acciones, estándar AlphaZero)
- Análisis de neuronas muertas y monitoreo de salud de la red
- Gestión de checkpoints con reanudación de entrenamiento
- Selección de movimientos basada en temperatura
- Estadísticas completas de partidas y logging en CSV

## 📊 Infraestructura de Entrenamiento
Optimizado para **GPUs Intel Arc** con precisión BFloat16, logrando entrenamiento eficiente mediante procesamiento por lotes (512 muestras) y generación paralela de autopartidas en los nodos de cómputo de Clementina.
