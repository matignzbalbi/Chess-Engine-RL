import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# Cargar variables de entorno
load_dotenv()

class ChessDatabase:
    """
    Conexión a Supabase PostgreSQL para guardar datos de entrenamiento.
    """
    
    def __init__(self):
        """Establece conexión con la base de datos"""
        try:
            # Opción 1: Usar URI completa
            if os.getenv("DATABASE_URL"):
                self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            
            # Opción 2: Usar parámetros separados
            else:
                self.conn = psycopg2.connect(
                    host=os.getenv("DB_HOST"),
                    database=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    port=os.getenv("DB_PORT", "6543")
                )
            
            print("✓ Conectado a Supabase PostgreSQL")
            
        except Exception as e:
            print(f"❌ Error al conectar con la base de datos: {e}")
            raise
    
    def test_connection(self):
        """Prueba la conexión consultando la tabla game_stats"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT COUNT(*) as total FROM game_stats")
                result = cur.fetchone()
                print(f"✓ Conexión exitosa. Registros en game_stats: {result['total']}") # type: ignore
                return True
        except Exception as e:
            print(f"❌ Error en test_connection: {e}")
            return False
    
    def insert_game_stat(self, iteration, game_id, stats):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO game_stats 
                    (iteration, game_id, total_moves, winner, termination_reason, unique_positions)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    iteration,
                    game_id,
                    stats['total_moves'],
                    stats['winner'],
                    stats['termination_reason'],
                    stats['unique_positions']
                ))
                
                self.conn.commit()
                print("✓ Registro insertado correctamente")
                
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error al insertar game_stat: {e}")
            raise

    
    def insert_game_stats_batch(self, data_list):
        """
        Inserta múltiples estadísticas de una vez (más eficiente).
        
        Args:
            data_list: lista de tuplas (iteration, game_id, total_moves, winner, 
                       termination_reason, unique_positions)
        """
        try:
            with self.conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO game_stats 
                    (iteration, game_id, total_moves, winner, termination_reason, unique_positions)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, data_list)
                
                self.conn.commit()
                print(f"✓ Insertados {len(data_list)} registros en game_stats")
                
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error al insertar batch: {e}")
            raise
    
    def get_iteration_summary(self, iteration):
        """Obtiene resumen de una iteración específica"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_games,
                        AVG(total_moves) as avg_moves,
                        SUM(CASE WHEN winner = 'white' THEN 1 ELSE 0 END) as white_wins,
                        SUM(CASE WHEN winner = 'black' THEN 1 ELSE 0 END) as black_wins,
                        SUM(CASE WHEN winner = 'draw' THEN 1 ELSE 0 END) as draws
                    FROM game_stats
                    WHERE iteration = %s
                """, (iteration,))
                
                return dict(cur.fetchone()) # type: ignore
                
        except Exception as e:
            print(f"❌ Error al obtener resumen: {e}")
            return None
    
    def close(self):
        """Cierra la conexión"""
        if self.conn:
            self.conn.close()
            print("✓ Conexión cerrada")