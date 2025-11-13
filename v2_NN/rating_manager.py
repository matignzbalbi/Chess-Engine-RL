
from datetime import datetime
import csv
import os

try:
    from glicko2 import Player as GlickoPlayer
    HAS_GLICKO = True
except Exception:
    HAS_GLICKO = False

# Fallback ELO simple
def expected_score_elo(Ra, Rb):
    from math import pow
    return 1.0 / (1.0 + pow(10.0, (Rb - Ra) / 400.0))

def update_elo(Ra, Rb, Sa, K=32):
    Ea = expected_score_elo(Ra, Rb)
    return Ra + K * (Sa - Ea)

class RatingManager:
    def __init__(self, use_glicko=True, storage_path="ratings.csv", initial_rating=1500):
        self.use_glicko = use_glicko and HAS_GLICKO
        self.initial_rating = initial_rating
        self.storage_path = storage_path

        self.players = {}  # checkpoint -> player object or dict for ELO
        # load existing ratings if present
        if os.path.exists(self.storage_path):
            self._load_storage()

    def _load_storage(self):
        with open(self.storage_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                ck = r["checkpoint"]
                if self.use_glicko:
                    p = GlickoPlayer(rating=float(r["rating"]), rd=float(r.get("rd", 350.0))) # type: ignore
                    self.players[ck] = p
                else:
                    self.players[ck] = {"rating": float(r["rating"]), "games": int(r.get("games_played", 0))}

    def ensure_player(self, checkpoint):
        if checkpoint in self.players:
            return
        if self.use_glicko:
            p = GlickoPlayer(rating=self.initial_rating)
            self.players[checkpoint] = p
        else:
            self.players[checkpoint] = {"rating": self.initial_rating, "games": 0}

    def record_match_result(self, a_ckpt, b_ckpt, score_a):

        self.ensure_player(a_ckpt)
        self.ensure_player(b_ckpt)

        if self.use_glicko:
            A = self.players[a_ckpt]
            B = self.players[b_ckpt]
            # glicko2 Player.update_player expects lists: opponent ratings, opponent rds, scores
            A.update_player([B.rating], [B.rd], [score_a])
            B.update_player([A.rating], [A.rd], [1.0 - score_a])
            # players updated in-place
        else:
            Ra = self.players[a_ckpt]["rating"]
            Rb = self.players[b_ckpt]["rating"]
            newRa = update_elo(Ra, Rb, score_a, K=32)
            newRb = update_elo(Rb, Ra, 1.0 - score_a, K=32)
            self.players[a_ckpt]["rating"] = newRa
            self.players[b_ckpt]["rating"] = newRb
            self.players[a_ckpt]["games"] += 1
            self.players[b_ckpt]["games"] += 1

    def get_rating(self, checkpoint):
        if checkpoint not in self.players:
            return None
        if self.use_glicko:
            p = self.players[checkpoint]
            return {"rating": p.rating, "rd": p.rd}
        else:
            p = self.players[checkpoint]
            return {"rating": p["rating"], "games": p["games"]}

    def save(self):
        # write current players to CSV
        fieldnames = ["checkpoint", "rating", "rd", "games_played", "timestamp"]
        rows = []
        for ck, p in self.players.items():
            if self.use_glicko:
                rows.append({
                    "checkpoint": ck,
                    "rating": float(p.rating),
                    "rd": float(p.rd),
                    "games_played": "",
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                rows.append({
                    "checkpoint": ck,
                    "rating": float(p["rating"]),
                    "rd": "",
                    "games_played": int(p.get("games", 0)),
                    "timestamp": datetime.utcnow().isoformat()
                })
        write_header = not os.path.exists(self.storage_path)
        with open(self.storage_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)