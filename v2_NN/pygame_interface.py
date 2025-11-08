"""
Interfaz Pygame para AlphaZero Chess
Adaptado a tu código con python-chess y MCTS

INSTALACIÓN:
pip install pygame chess torch

ESTRUCTURA DE ARCHIVOS:
proyecto/
  ├── pygame_interface.py (este archivo)
  ├── chess_game.py
  ├── model.py
  ├── mcts.py
  ├── move_mapping.py
  └── pytorch_files/
      └── model_X.pt

USO:
python pygame_interface.py
"""

import pygame
import chess
import torch
import time
import sys
from pathlib import Path

# Importar tus módulos
from chess_game import ChessGame
from model import create_chess_model
from mcts import MCTS


# ============================================================================
# CONSTANTES
# ============================================================================

# Dimensiones
WIDTH = 1000
HEIGHT = 720
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
INFO_PANEL_WIDTH = WIDTH - BOARD_SIZE - 20
INFO_PANEL_X = BOARD_SIZE + 20

# Colores
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
SELECTED_COLOR = (246, 246, 105)
LAST_MOVE_COLOR = (205, 210, 106)
BG_COLOR = (49, 46, 43)
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)

# Configuración del juego
FPS = 60


# ============================================================================
# RENDERIZADO DE PIEZAS
# ============================================================================

def create_piece_surfaces():
    """Crea superficies de Pygame con símbolos de piezas usando fuentes Unicode"""
    pieces = {}
    
    # Símbolos Unicode de piezas de ajedrez
    piece_unicode = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # Blancas
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'   # Negras
    }
    
    # Intentar cargar fuentes con símbolos de ajedrez
    font_size = int(SQUARE_SIZE * 0.8)
    
    # Lista de fuentes que tienen símbolos de ajedrez
    font_names = [
        'segoeuisymbol',  # Windows
        'dejavusans',     # Linux
        'Arial Unicode MS',  # Mac
        'freesans',       # Linux alternativo
        None              # Fuente por defecto de Pygame
    ]
    
    font = None
    for font_name in font_names:
        try:
            if font_name:
                font = pygame.font.SysFont(font_name, font_size)
            else:
                font = pygame.font.Font(None, font_size)
            # Probar si puede renderizar símbolos
            test = font.render('♔', True, (0, 0, 0))
            break
        except:
            continue
    
    if font is None:
        font = pygame.font.Font(None, font_size)
    
    # Crear superficie para cada pieza
    for symbol, unicode_char in piece_unicode.items():
        # Determinar color
        color = (255, 255, 255) if symbol.isupper() else (0, 0, 0)
        
        # Renderizar el símbolo
        text_surface = font.render(unicode_char, True, color)
        
        # Crear superficie del tamaño de la casilla
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        
        # Centrar el símbolo
        text_rect = text_surface.get_rect(center=(SQUARE_SIZE // 2, SQUARE_SIZE // 2))
        surface.blit(text_surface, text_rect)
        
        # Agregar borde/sombra para mejor visibilidad
        outline_color = (0, 0, 0) if symbol.isupper() else (255, 255, 255)
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            outline = font.render(unicode_char, True, outline_color)
            outline_rect = outline.get_rect(center=(SQUARE_SIZE // 2 + dx, SQUARE_SIZE // 2 + dy))
            temp_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            temp_surface.blit(outline, outline_rect)
            surface.blit(temp_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        
        # Re-dibujar símbolo principal encima
        surface.blit(text_surface, text_rect)
        
        pieces[symbol] = surface
    
    return pieces


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class ChessGUI:
    """Interfaz gráfica principal para el ajedrez"""
    
    def __init__(self, model_path, num_resBlocks=2, num_hidden=32, num_searches=100):
        """Inicializa la GUI"""
        pygame.init()
        
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AlphaZero Chess - Humano vs IA")
        self.clock = pygame.time.Clock()
        
        # Fuentes
        self.font_title = pygame.font.SysFont('Arial', 32, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_medium = pygame.font.SysFont('Arial', 18)
        self.font_small = pygame.font.SysFont('Arial', 14)
        
        # Estado del juego
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.human_color = chess.WHITE
        self.game_over = False
        self.ai_thinking = False
        
        # Historial
        self.move_history = []
        self.ai_stats = {
            'confidence': 0.0,
            'think_time': 0.0,
            'evaluations': 0
        }
        
        # Cargar imágenes de piezas
        print("Generando gráficos de piezas...")
        self.piece_images = create_piece_surfaces()
        print("✓ Piezas renderizadas")
        
        # Cargar modelo
        print(f"\nCargando modelo desde {model_path}...")
        self.game = ChessGame()
        self.model = create_chess_model(self.game, num_resBlocks, num_hidden)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print("✓ Modelo cargado exitosamente")
        except FileNotFoundError:
            print(f"❌ Error: No se encontró {model_path}")
            sys.exit(1)
        
        # Configurar MCTS
        self.args = {'C': 2, 'num_searches': num_searches}
        self.mcts = MCTS(self.game, self.args, self.model)
        print(f"✓ MCTS configurado con {num_searches} búsquedas\n")
        
        # Botones
        self.buttons = self._create_buttons()
        
        print("="*60)
        print("CONTROLES:")
        print("  - Click en pieza para seleccionar")
        print("  - Click en casilla válida para mover")
        print("  - Botones en panel derecho para controles")
        print("="*60 + "\n")
    
    def _create_buttons(self):
        """Crea botones de control"""
        buttons = {}
        button_width = INFO_PANEL_WIDTH - 20
        button_height = 40
        x = INFO_PANEL_X + 10
        y_start = 500
        spacing = 50
        
        button_labels = ['Nueva Partida', 'Deshacer', 'Girar Tablero']
        
        for i, label in enumerate(button_labels):
            buttons[label] = {
                'rect': pygame.Rect(x, y_start + i * spacing, button_width, button_height),
                'label': label
            }
        
        return buttons
    
    def _draw_board(self):
        """Dibuja el tablero de ajedrez"""
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
        
        # Dibujar coordenadas
        for i in range(8):
            # Números (filas)
            text = self.font_small.render(str(8 - i), True, DARK_SQUARE if i % 2 == 0 else LIGHT_SQUARE)
            self.screen.blit(text, (5, i * SQUARE_SIZE + 5))
            
            # Letras (columnas)
            text = self.font_small.render(chr(97 + i), True, LIGHT_SQUARE if i % 2 == 1 else DARK_SQUARE)
            self.screen.blit(text, (i * SQUARE_SIZE + SQUARE_SIZE - 15, BOARD_SIZE - 20))
    
    def _draw_highlights(self):
        """Dibuja highlights de casillas"""
        # Último movimiento
        if self.last_move:
            for square in [self.last_move.from_square, self.last_move.to_square]:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                s.set_alpha(128)
                s.fill(LAST_MOVE_COLOR)
                self.screen.blit(s, rect)
        
        # Casilla seleccionada
        if self.selected_square is not None:
            row = 7 - chess.square_rank(self.selected_square)
            col = chess.square_file(self.selected_square)
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(180)
            s.fill(SELECTED_COLOR)
            self.screen.blit(s, rect)
            
            # Movimientos legales
            for move in self.legal_moves:
                row = 7 - chess.square_rank(move.to_square)
                col = chess.square_file(move.to_square)
                center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                
                # Círculo para casillas vacías, anillo para capturas
                if self.board.piece_at(move.to_square):
                    pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, center, SQUARE_SIZE // 2 - 5, 5)
                else:
                    pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, center, 12)
    
    def _draw_pieces(self):
        """Dibuja las piezas en el tablero"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                
                # Obtener superficie de la pieza
                piece_surface = self.piece_images.get(piece.symbol())
                if piece_surface:
                    x = col * SQUARE_SIZE
                    y = row * SQUARE_SIZE
                    self.screen.blit(piece_surface, (x, y))
    
    def _draw_info_panel(self):
        """Dibuja el panel de información"""
        panel_x = INFO_PANEL_X
        y = 20
        
        # Título
        title = self.font_title.render("AlphaZero", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x + 40, y))
        y += 60
        
        # Estado del turno
        if self.game_over:
            if self.board.is_checkmate():
                winner = "Negras" if self.board.turn == chess.WHITE else "Blancas"
                status = f"¡{winner} ganan!"
                color = (255, 215, 0)
            else:
                status = "Empate"
                color = (200, 200, 200)
        elif self.ai_thinking:
            status = "IA pensando..."
            color = (255, 165, 0)
        else:
            status = "Tu turno"
            color = (100, 255, 100)
        
        status_text = self.font_large.render(status, True, color)
        self.screen.blit(status_text, (panel_x, y))
        y += 50
        
        # Información de la partida
        info_items = [
            ("Movimiento:", str(len(self.board.move_stack))),
            ("Jugador:", "Blancas" if self.human_color == chess.WHITE else "Negras"),
            ("", ""),  # Espacio
            ("Última IA:", ""),
            ("  Confianza:", f"{self.ai_stats['confidence']:.1%}" if self.ai_stats['confidence'] > 0 else "-"),
            ("  Tiempo:", f"{self.ai_stats['think_time']:.1f}s" if self.ai_stats['think_time'] > 0 else "-"),
        ]
        
        for label, value in info_items:
            if label == "":
                y += 10
                continue
            
            text = self.font_medium.render(label, True, (200, 200, 200))
            self.screen.blit(text, (panel_x, y))
            
            if value:
                val_text = self.font_medium.render(value, True, TEXT_COLOR)
                self.screen.blit(val_text, (panel_x + 150, y))
            
            y += 30
        
        # Historial de movimientos
        y += 20
        hist_title = self.font_large.render("Historial", True, TEXT_COLOR)
        self.screen.blit(hist_title, (panel_x, y))
        y += 40
        
        # Mostrar últimos 10 movimientos
        start_idx = max(0, len(self.move_history) - 10)
        for i in range(start_idx, len(self.move_history)):
            move_text = self.font_small.render(self.move_history[i], True, (220, 220, 220))
            self.screen.blit(move_text, (panel_x + 10, y))
            y += 22
    
    def _draw_buttons(self):
        """Dibuja botones de control"""
        mouse_pos = pygame.mouse.get_pos()
        
        for button_data in self.buttons.values():
            rect = button_data['rect']
            label = button_data['label']
            
            # Color según hover
            color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
            
            # Dibujar botón
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            
            # Texto
            text = self.font_medium.render(label, True, TEXT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def _draw_thinking_indicator(self):
        """Dibuja indicador de pensamiento de IA"""
        if not self.ai_thinking:
            return
        
        # Puntos animados
        center_x = INFO_PANEL_X + INFO_PANEL_WIDTH // 2
        center_y = 180
        num_dots = 8
        radius = 20
        
        for i in range(num_dots):
            angle = (pygame.time.get_ticks() / 100 + i * 45) % 360
            angle_rad = angle * 3.14159 / 180
            
            x = int(center_x + radius * pygame.math.Vector2(1, 0).rotate(angle).x)
            y = int(center_y + radius * pygame.math.Vector2(1, 0).rotate(angle).y)
            
            # Fade basado en posición
            alpha = int(128 + 127 * ((i / num_dots)))
            color = (255, 165, 0)
            
            pygame.draw.circle(self.screen, color, (x, y), 6)
    
    def _get_square_under_mouse(self):
        """Obtiene la casilla bajo el cursor del mouse"""
        mouse_pos = pygame.mouse.get_pos()
        x, y = mouse_pos
        
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return None
        
        col = x // SQUARE_SIZE
        row = 7 - (y // SQUARE_SIZE)
        
        return chess.square(col, row)
    
    def _handle_square_click(self, square):
        """Maneja el click en una casilla del tablero"""
        if self.game_over or self.ai_thinking or self.board.turn != self.human_color:
            return
        
        # Si no hay casilla seleccionada
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                self.selected_square = square
                self.legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
        
        # Si hay casilla seleccionada, intentar mover
        else:
            move = None
            for m in self.legal_moves:
                if m.to_square == square:
                    move = m
                    break
            
            if move:
                # Aplicar movimiento
                self.board.push(move)
                self.last_move = move
                
                # Agregar al historial
                move_num = (len(self.board.move_stack) + 1) // 2
                move_str = f"{move_num}. {move.uci()}"
                self.move_history.append(move_str)
                
                # Verificar fin de juego
                if self.board.is_game_over():
                    self.game_over = True
                
                # Limpiar selección
                self.selected_square = None
                self.legal_moves = []
            
            else:
                # Deseleccionar o seleccionar otra pieza
                piece = self.board.piece_at(square)
                if piece and piece.color == self.human_color:
                    self.selected_square = square
                    self.legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
                else:
                    self.selected_square = None
                    self.legal_moves = []
    
    def _handle_button_click(self, pos):
        """Maneja clicks en botones"""
        for label, button_data in self.buttons.items():
            if button_data['rect'].collidepoint(pos):
                if label == 'Nueva Partida':
                    self._new_game()
                elif label == 'Deshacer':
                    self._undo_move()
                elif label == 'Girar Tablero':
                    self._flip_board()
                return True
        return False
    
    def _new_game(self):
        """Inicia nueva partida"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.game_over = False
        self.ai_thinking = False
        self.move_history = []
        self.ai_stats = {'confidence': 0.0, 'think_time': 0.0, 'evaluations': 0}
    
    def _undo_move(self):
        """Deshace los últimos 2 movimientos (humano + IA)"""
        if len(self.board.move_stack) >= 2 and not self.ai_thinking:
            self.board.pop()
            self.board.pop()
            self.move_history = self.move_history[:-2]
            self.game_over = False
            self.last_move = self.board.peek() if self.board.move_stack else None
    
    def _flip_board(self):
        """Gira el tablero (cambia el color del jugador)"""
        self.human_color = not self.human_color
    
    def _ai_move(self):
        """La IA hace su movimiento"""
        if self.game_over or self.board.turn == self.human_color or self.ai_thinking:
            return
        
        self.ai_thinking = True
        
        # Ejecutar MCTS
        start_time = time.time()
        action_probs = self.mcts.search(self.board)
        
        # Obtener mejor movimiento
        action = action_probs.argmax()
        move = self.game.get_move_from_action(self.board, action)
        
        # Registrar stats
        self.ai_stats['think_time'] = time.time() - start_time
        self.ai_stats['confidence'] = float(action_probs[action])
        
        # Aplicar movimiento
        self.board.push(move)
        self.last_move = move
        
        # Agregar al historial
        move_num = (len(self.board.move_stack) + 1) // 2
        move_str = f"{move_num}. {move.uci()} (IA)"
        self.move_history.append(move_str)
        
        print(f"IA jugó: {move.uci()} | Confianza: {self.ai_stats['confidence']:.1%} | Tiempo: {self.ai_stats['think_time']:.1f}s")
        
        # Verificar fin de juego
        if self.board.is_game_over():
            self.game_over = True
        
        self.ai_thinking = False
    
    def run(self):
        """Loop principal del juego"""
        running = True
        
        while running:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    
                    # Verificar clicks en botones
                    if self._handle_button_click(pos):
                        continue
                    
                    # Click en tablero
                    square = self._get_square_under_mouse()
                    if square is not None:
                        self._handle_square_click(square)
            
            # Movimiento de IA
            if not self.game_over and self.board.turn != self.human_color and not self.ai_thinking:
                self._ai_move()
            
            # Renderizado
            self.screen.fill(BG_COLOR)
            self._draw_board()
            self._draw_highlights()
            self._draw_pieces()
            self._draw_info_panel()
            self._draw_buttons()
            self._draw_thinking_indicator()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

def main():
    """Función principal"""
    print("\n" + "="*60)
    print("ALPHAZERO CHESS - PYGAME INTERFACE")
    print("="*60 + "\n")
    
    # Configuración
    MODEL_PATH = "pytorch_files/model_4.pt"
    NUM_RESBLOCKS = 2
    NUM_HIDDEN = 32
    NUM_SEARCHES = 100
    
    # Verificar si existe el modelo
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: No se encontró el modelo en '{MODEL_PATH}'")
        print("\nAsegúrate de:")
        print("  1. Haber entrenado un modelo")
        print("  2. Ajustar MODEL_PATH con la ruta correcta")
        sys.exit(1)
    
    # Ajustar búsquedas según hardware
    if not torch.cuda.is_available():
        print("⚠️  GPU no detectada. Reduciendo búsquedas MCTS para mejor rendimiento")
        NUM_SEARCHES = 50
    
    # Crear e iniciar GUI
    try:
        gui = ChessGUI(
            model_path=MODEL_PATH,
            num_resBlocks=NUM_RESBLOCKS,
            num_hidden=NUM_HIDDEN,
            num_searches=NUM_SEARCHES
        )
        
        print("✓ GUI lista. Iniciando juego...\n")
        gui.run()
        
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()