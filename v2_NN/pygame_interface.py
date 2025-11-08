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
    """Carga imágenes de piezas desde la carpeta assets/pieces"""
    pieces = {}
    piece_filenames = {
        'K': 'wK.png', 'Q': 'wQ.png', 'R': 'wR.png', 'B': 'wB.png', 'N': 'wN.png', 'P': 'wP.png',
        'k': 'bK.png', 'q': 'bQ.png', 'r': 'bR.png', 'b': 'bB.png', 'n': 'bN.png', 'p': 'bP.png'
    }

    for symbol, filename in piece_filenames.items():
        image_path = Path("assets/pieces") / filename
        if not image_path.exists():
            print(f"⚠️ No se encontró {image_path}, se omitirá {symbol}")
            continue

        # Cargar imagen con transparencia
        image = pygame.image.load(str(image_path)).convert_alpha()
        image = pygame.transform.smoothscale(image, (SQUARE_SIZE, SQUARE_SIZE))
        pieces[symbol] = image

    if not pieces:
        print("❌ No se cargó ninguna imagen de pieza. Verificá la carpeta assets/pieces/")
    else:
        print(f"✓ {len(pieces)} piezas cargadas correctamente.")

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
        pygame.display.set_caption("AlphaZero Chess - Humano vs Bot RL")
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
        self.flipped = False  # False = blancas abajo, True = negras abajo
        self.game_over = False
        self.ai_thinking = False
        
        # Historial
        self.move_history = []
        self.ai_stats = {
            'confidence': 0.0,
            'think_time': 0.0,
            'evaluations': 0
        }
        self.scroll_offset = 0
        self.scroll_speed = 20
        self.max_scroll = 0

        
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
        """Define etiquetas de botones (ya no define posiciones fijas)."""
        return {
            'Nueva Partida': {},
            'Deshacer': {},
            'Girar Tablero': {}
        }
    
    def _draw_board(self):
        """Dibuja el tablero de ajedrez (compatible con giro)."""
        for row in range(8):
            for col in range(8):
                # Calcular coordenadas según orientación
                draw_row = row
                draw_col = col
                if self.flipped:
                    draw_row = 7 - row
                    draw_col = 7 - col

                # Color del cuadrado
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                rect = pygame.Rect(draw_col * SQUARE_SIZE, draw_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        # Dibujar letras y números del borde
        for i in range(8):
            # Números (filas)
            row_label = str(i + 1) if self.flipped else str(8 - i)
            text = self.font_small.render(row_label, True, (60, 60, 60))
            self.screen.blit(text, (5, i * SQUARE_SIZE + 5))

            # Letras (columnas)
            col_label = chr(104 - i) if self.flipped else chr(97 + i)
            text = self.font_small.render(col_label, True, (60, 60, 60))
            self.screen.blit(text, (i * SQUARE_SIZE + SQUARE_SIZE - 20, BOARD_SIZE - 20))


    
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
        """Dibuja las piezas en el tablero, respetando la orientación (flip)."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_surface = self.piece_images.get(piece.symbol())
                if not piece_surface:
                    continue

                # Obtener fila y columna de la pieza en el tablero lógico
                row = chess.square_rank(square)
                col = chess.square_file(square)

                # Si el tablero está girado, invertir coordenadas
                if self.flipped:
                    row = row
                    col = 7 - col
                else:
                    row = 7 - row
                    col = col

                # Calcular posición en pantalla
                x = col * SQUARE_SIZE
                y = row * SQUARE_SIZE

                # Dibujar la pieza en la posición correspondiente
                self.screen.blit(piece_surface, (x, y))

    
    def _draw_info_panel(self):
        """Dibuja el panel de información con historial a la izquierda y botones a la derecha."""
        panel_x = INFO_PANEL_X
        y = 20

        # ======================
        # Título
        # ======================
        title = self.font_title.render("AlphaZero", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x + 40, y))
        y += 60

        # ======================
        # Estado del juego
        # ======================
        if self.game_over:
            if self.board.is_checkmate():
                winner = "Negras" if self.board.turn == chess.WHITE else "Blancas"
                status = f"¡{winner} ganan!"
                color = (255, 215, 0)
            else:
                status = "Empate"
                color = (200, 200, 200)
        elif self.ai_thinking:
            status = "Bot RL pensando..."
            color = (255, 165, 0)
        else:
            status = "Tu turno"
            color = (100, 255, 100)

        status_text = self.font_large.render(status, True, color)
        self.screen.blit(status_text, (panel_x, y))
        y += 50

        # ======================
        # Datos generales
        # ======================
        info_items = [
            ("Movimiento:", str(len(self.board.move_stack))),
            ("Jugador:", "Blancas" if self.human_color == chess.WHITE else "Negras"),
            ("Última Bot RL:", ""),
            ("  Confianza:", f"{self.ai_stats['confidence']:.1%}" if self.ai_stats['confidence'] > 0 else "-"),
            ("  Tiempo:", f"{self.ai_stats['think_time']:.1f}s" if self.ai_stats['think_time'] > 0 else "-"),
        ]

        for label, value in info_items:
            text = self.font_medium.render(label, True, (200, 200, 200))
            self.screen.blit(text, (panel_x, y))
            val_text = self.font_medium.render(value, True, TEXT_COLOR)
            self.screen.blit(val_text, (panel_x + 150, y))
            y += 28

        # ======================
        # División del panel: historial + botones
        # ======================
        y += 20
        hist_title = self.font_large.render("Historial", True, TEXT_COLOR)
        self.screen.blit(hist_title, (panel_x, y))
        y += 40

        # Mitad del ancho del panel
        half_width = (INFO_PANEL_WIDTH - 20) // 2

        # ---- HISTORIAL ----
        history_surface_height = 250
        history_surface = pygame.Surface((half_width - 10, history_surface_height))
        history_surface.fill((40, 40, 40))

        y_offset = 10 - self.scroll_offset
        for move_str in self.move_history:
            move_text = self.font_small.render(move_str, True, (220, 220, 220))
            history_surface.blit(move_text, (10, y_offset))
            y_offset += 22

        total_height = len(self.move_history) * 22 + 10
        self.max_scroll = max(0, total_height - history_surface_height)
        self.screen.blit(history_surface, (panel_x, y))

        # Scrollbar visual
        if self.max_scroll > 0:
            scrollbar_height = int(history_surface_height * (history_surface_height / total_height))
            scrollbar_y = y + int((self.scroll_offset / self.max_scroll) * (history_surface_height - scrollbar_height))
            scrollbar_rect = pygame.Rect(panel_x + half_width - 15, scrollbar_y, 6, scrollbar_height)
            pygame.draw.rect(self.screen, (160, 160, 160), scrollbar_rect)

        # ---- BOTONES (columna derecha) ----
        button_x = panel_x + half_width + 10
        button_y = y
        button_w = half_width - 10
        button_h = 45
        spacing = 55

        for label in ['Nueva Partida', 'Deshacer', 'Girar Tablero']:
            rect = pygame.Rect(button_x, button_y, button_w, button_h)
            pygame.draw.rect(self.screen, BUTTON_COLOR, rect, border_radius=8)
            text = self.font_medium.render(label, True, TEXT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
            button_y += spacing
            self.buttons[label]['rect'] = rect  # ✅ Actualiza posición real del botón


        # (No necesitamos guardar estos botones porque son redibujados aquí)

    
    def _draw_thinking_indicator(self):
        """Dibuja indicador de pensamiento de Bot RL"""
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
        """Obtiene la casilla bajo el cursor del mouse considerando la orientación."""
        mouse_pos = pygame.mouse.get_pos()
        x, y = mouse_pos

        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return None

        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE

        if self.flipped:
            col = 7 - col
            row = 7 - row
        else:
            row = 7 - row

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
        """Deshace los últimos 2 movimientos (humano + Bot RL)"""
        if len(self.board.move_stack) >= 2 and not self.ai_thinking:
            self.board.pop()
            self.board.pop()
            self.move_history = self.move_history[:-2]
            self.game_over = False
            self.last_move = self.board.peek() if self.board.move_stack else None
    
    def _flip_board(self):
        """Gira el tablero (cambia la orientación visual)"""
        self.flipped = not self.flipped
        print("↻ Tablero girado:", "negras abajo" if self.flipped else "blancas abajo")

    
    def _ai_move(self):
        """La Bot RL hace su movimiento"""
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
        move_str = f"{move_num}. {move.uci()} (Bot RL)"
        self.move_history.append(move_str)
        
        print(f"Bot RL jugó: {move.uci()} | Confianza: {self.ai_stats['confidence']:.1%} | Tiempo: {self.ai_stats['think_time']:.1f}s")
        
        # Verificar fin de juego
        if self.board.is_game_over():
            self.game_over = True
        
        self.ai_thinking = False
    
    def run(self):
        """Loop principal del juego"""
        running = True
        self.fullscreen = False  # Por si no estaba en __init__

        while running:
            # ==========================
            # 🎮 EVENTOS
            # ==========================
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # 🔁 Alternar pantalla completa (F11)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self.fullscreen = not self.fullscreen
                        if self.fullscreen:
                            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

                # 🖱️ Scroll con la rueda del mouse
                elif event.type == pygame.MOUSEWHEEL:
                    # event.y = 1 (rueda arriba) o -1 (rueda abajo)
                    self.scroll_offset -= event.y * self.scroll_speed
                    self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))

                # 🖱️ Click del mouse
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    # Verificar clicks en botones del panel lateral
                    if self._handle_button_click(pos):
                        continue

                    # Click en el tablero de ajedrez
                    square = self._get_square_under_mouse()
                    if square is not None:
                        self._handle_square_click(square)

            # ==========================
            # 🧠 Movimiento de la Bot RL
            # ==========================
            if not self.game_over and self.board.turn != self.human_color and not self.ai_thinking:
                self._ai_move()

            # ==========================
            # 🖼️ RENDERIZADO
            # ==========================
            self.screen.fill(BG_COLOR)
            self._draw_board()
            self._draw_highlights()
            self._draw_pieces()
            self._draw_info_panel()
            self._draw_thinking_indicator()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()



# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

# ============================================================================
# MENÚ DE DIFICULTAD
# ============================================================================

def difficulty_menu():
    """Muestra una pantalla inicial para elegir dificultad."""
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Seleccionar dificultad - Bot RL")
    font_title = pygame.font.SysFont('Arial', 36, bold=True)
    font_button = pygame.font.SysFont('Arial', 24, bold=True)
    clock = pygame.time.Clock()

    # Colores
    BG_COLOR = (30, 30, 30)
    BUTTON_COLOR = (70, 130, 180)
    BUTTON_HOVER = (100, 160, 210)
    TEXT_COLOR = (255, 255, 255)

    # Opciones de dificultad
    buttons = {
        "Principiante": {
            "rect": pygame.Rect(180, 130, 240, 50),
            "model": "pytorch_files/model_1.pt"
        },
        "Intermedio": {
            "rect": pygame.Rect(180, 200, 240, 50),
            "model": "pytorch_files/model_2.pt"
        },
        "Avanzado": {
            "rect": pygame.Rect(180, 270, 240, 50),
            "model": "pytorch_files/model_3.pt"
        }
    }

    selected_model = None

    # Loop del menú
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for label, data in buttons.items():
                    if data["rect"].collidepoint(pos):
                        selected_model = data["model"]
                        return selected_model

        # Dibujar fondo
        screen.fill(BG_COLOR)

        # Título
        title = font_title.render("Seleccioná la dificultad", True, TEXT_COLOR)
        screen.blit(title, (110, 60))

        # Dibujar botones
        mouse_pos = pygame.mouse.get_pos()
        for label, data in buttons.items():
            rect = data["rect"]
            color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(screen, color, rect, border_radius=10)
            text = font_button.render(label, True, TEXT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)

def main():
    """Función principal"""
    print("\n" + "="*60)
    print("BOT RL CHESS - PYGAME INTERFACE")
    print("="*60 + "\n")

    # 1️⃣ Elegir dificultad
    model_path = difficulty_menu()

    # 2️⃣ Parámetros de red según modelo
    NUM_RESBLOCKS = 2
    NUM_HIDDEN = 32
    NUM_SEARCHES = 100

    # 3️⃣ Verificar modelo seleccionado
    if not Path(model_path).exists():
        print(f"❌ Error: No se encontró el modelo en '{model_path}'")
        sys.exit(1)

    # 4️⃣ Ajustar búsquedas según hardware
    if not torch.cuda.is_available():
        print("⚠️ GPU no detectada. Reduciendo búsquedas MCTS para mejor rendimiento")
        NUM_SEARCHES = 50

    # 5️⃣ Crear e iniciar GUI
    try:
        gui = ChessGUI(
            model_path=model_path,
            num_resBlocks=NUM_RESBLOCKS,
            num_hidden=NUM_HIDDEN,
            num_searches=NUM_SEARCHES
        )
        print(f"✓ Dificultad seleccionada: {Path(model_path).name.replace('.pt','').replace('_',' ').capitalize()}")
        print("✓ GUI lista. Iniciando juego...\n")
        gui.run()

    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()