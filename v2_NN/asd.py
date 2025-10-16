import csv
from move_mapping import MoveMapper  # o del archivo donde está tu clase

# Crear el objeto y generar los mapeos
mapper = MoveMapper()

# === Exportar _move_to_index ===
with open("move_to_index.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["from_square", "to_square", "promotion", "action_index"])
    for (from_sq, to_sq, promo), idx in mapper._move_to_index.items():
        writer.writerow([from_sq, to_sq, promo, idx])

# === Exportar _index_to_move ===
with open("index_to_move.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["action_index", "from_square", "to_square", "promotion"])
    for idx, (from_sq, to_sq, promo) in mapper._index_to_move.items():
        writer.writerow([idx, from_sq, to_sq, promo])

print("✓ Archivos CSV generados correctamente:")
print(" - move_to_index.csv")
print(" - index_to_move.csv")
