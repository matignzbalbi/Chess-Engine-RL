import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from model import ChessResNet
from move_mapping import MoveMapper
from chess_game import ChessGame

# =====================================================
# 1) Captura de activaciones con forward hooks
# =====================================================
def capturar_activaciones(modelo):
    activations = {}

    def hook_fn(nombre):
        def hook(modulo, input, output):
            if isinstance(output, torch.Tensor):
                activations[nombre] = output.detach().cpu().flatten().numpy()
            elif isinstance(output, (list, tuple)):
                activations[nombre] = torch.cat([o.flatten() for o in output]).detach().cpu().numpy()
        return hook

    handles = []
    for nombre, modulo in modelo.named_modules():
        if len(list(modulo.children())) == 0:  # solo capas hojas
            h = modulo.register_forward_hook(hook_fn(nombre))
            handles.append(h)

    return activations, handles

# =====================================================
# 2) Ejecutar forward y obtener activaciones
# =====================================================
def obtener_activaciones(modelo, input_tensor):
    modelo.eval()
    activaciones, handles = capturar_activaciones(modelo)

    with torch.no_grad():
        modelo(input_tensor)

    for h in handles:
        h.remove()

    return activaciones

# =====================================================
# 3) Comparar activaciones entre modelos
# =====================================================
def comparar_activaciones(act_A, act_B):
    comunes = set(act_A.keys()) & set(act_B.keys())
    diferencias = {}

    for capa in comunes:
        vA = act_A[capa]
        vB = act_B[capa]

        # Igualamos longitudes
        m = min(len(vA), len(vB))
        if m == 0:
            continue

        diff = ((vA[:m] - vB[:m]) ** 2).mean()
        diferencias[capa] = diff

    return diferencias

# =====================================================
# 4) Guardar gráficos
# =====================================================
def plot_activaciones(act_A, act_B, out_folder, capa):
    vA = act_A[capa]
    vB = act_B[capa]

    m = min(len(vA), len(vB))
    vA = vA[:m]
    vB = vB[:m]

    plt.figure(figsize=(10, 4))
    plt.plot(vA, label="Modelo A", alpha=0.7)
    plt.plot(vB, label="Modelo B", alpha=0.7)
    plt.title(f"Activaciones – {capa}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_folder, f"{capa.replace('/', '_')}.png"))
    plt.close()

# =====================================================
# 5) MAIN
# =====================================================
if __name__ == "__main__":
    logging.info("Cargando modelos...")

    # Rutas
    RUTA_MODELO_A = r"C:\Users\mrgnz\OneDrive\Escritorio\AI-MTCS-NN\pytorch_files\model_0.pt"
    RUTA_CONFIG_A = r"C:\Users\mrgnz\OneDrive\Escritorio\AI-MTCS-NN\pytorch_files\model_0_config.json"

    # Si solo vas a comparar un modelo, podés repetir:
    RUTA_MODELO_B = r"C:\Users\mrgnz\OneDrive\Escritorio\AI-MTCS-NN\pytorch_files\model_0.pt"
    RUTA_CONFIG_B = r"C:\Users\mrgnz\OneDrive\Escritorio\AI-MTCS-NN\pytorch_files\model_0_config.json"


    # Cargar configs
    def load_config(path):
        with open(path, "r") as f:
            cfg = json.load(f)
        logging.info(f"Cargando config JSON: num_resBlocks={cfg['num_resBlocks']}, num_hidden={cfg['num_hidden']}, action_size={cfg['action_size']}")
        return cfg

    cfg_A = load_config(RUTA_CONFIG_A)
    cfg_B = load_config(RUTA_CONFIG_B)

    # Instanciar game y mapeo de movimientos
    game = ChessGame()

    # Crear mapper
    mapper = MoveMapper(include_queen_promotions=False)

    # Obtener tamaño real
    action_size = mapper.action_size

    logging.info("Action size:", action_size)


    # Crear modelos
    model_A = ChessResNet(game, cfg_A["num_resBlocks"], cfg_A["num_hidden"])
    model_B = ChessResNet(game, cfg_B["num_resBlocks"], cfg_B["num_hidden"])

    # Cargar pesos
    model_A.load_state_dict(torch.load(RUTA_MODELO_A, map_location="cpu"))
    model_B.load_state_dict(torch.load(RUTA_MODELO_B, map_location="cpu"))

    logging.info("Modelos cargados correctamente.")

    # Input de prueba: tablero vacío (o como prefieras)
    dummy_input = torch.zeros((1, 12, 8, 8))

    logging.info("Instrumentando y ejecutando forward...")
    act_A = obtener_activaciones(model_A, dummy_input)
    act_B = obtener_activaciones(model_B, dummy_input)

    capas_A = set(act_A.keys())
    capas_B = set(act_B.keys())
    comunes = capas_A & capas_B

    logging.info(f"Capas A: {len(capas_A)}, Capas B: {len(capas_B)}, Comunes: {len(comunes)}")
    logging.info("Ejemplos capas:", list(capas_A)[:20])

    # Comparación
    diffs = comparar_activaciones(act_A, act_B)

    # Carpeta output
    out_folder = os.path.join("activation_comparison_plots", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_folder, exist_ok=True)

    # Guardar CSV resumen
    csv_path = os.path.join(out_folder, "activation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("capa,mse\n")
        for capa, mse in diffs.items():
            f.write(f"{capa},{mse}\n")

    logging.info(f"Guardado resumen CSV en: {csv_path}")

    # Plots
    for capa in diffs.keys():
        plot_activaciones(act_A, act_B, out_folder, capa)

    logging.info("Plots guardados en:", out_folder)
    logging.info("Comparación completada.")