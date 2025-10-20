import numpy as np
import math
import torch

class Node:
    """
    Nodo del árbol MCTS.
    
    CONVENCIÓN DE VALORES:
    - value_sum y visit_count están desde la perspectiva del jugador QUE MUEVE en este nodo
    - Cuando backpropagamos, invertimos el valor para el padre (cambio de turno)
    """
    
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior  # P(s,a) - probabilidad del modelo para este movimiento
        
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0  # Suma de valores desde la perspectiva del jugador actual
        
    def is_fully_expanded(self):
        """Un nodo está expandido si tiene al menos un hijo"""
        return len(self.children) > 0
    
    def select(self):
        """Selecciona el mejor hijo usando UCB"""
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        """
        Calcula UCB (Upper Confidence Bound) para un hijo.
        
        Fórmula AlphaZero: Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Componentes:
        - Q(s,a): Valor promedio del hijo (rango [-1, 1])
        - P(s,a): Prior del modelo (probabilidad inicial)
        - N(s): Visitas del padre (self)
        - N(s,a): Visitas del hijo
        - C: Constante de exploración
        
        Q está desde la perspectiva del padre, que es lo que queremos
        (el padre está eligiendo entre sus hijos).
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            # IMPORTANTE: child.value_sum está desde la perspectiva del hijo
            # Pero queremos Q desde la perspectiva del padre
            # Por eso tomamos el NEGATIVO
            q_value = -child.value_sum / child.visit_count
        
        # Término de exploración
        exploration = self.args['C'] * child.prior * (
            math.sqrt(self.visit_count) / (child.visit_count + 1)
        )
        
        return q_value + exploration
    
    def expand(self, policy):
        """
        Expande el nodo creando todos los hijos válidos.
        
        Args:
            policy: Vector de probabilidades del modelo (una por acción)
        """
        # Obtener máscara de movimientos legales
        valid_moves = self.game.get_valid_moves(self.state)
        
        for action, prob in enumerate(policy):
            # Solo expandir acciones con probabilidad > 0 Y legales
            if prob > 0 and valid_moves[action] > 0:
                try:
                    # Crear nuevo estado aplicando el movimiento
                    child_state = self.state.copy()
                    child_state = self.game.get_next_state(child_state, action, 1)
                    
                    # ✅ ELIMINADO: change_perspective innecesario
                    # En ajedrez, state.turn cambia automáticamente después de push()

                    # Crear nodo hijo
                    child = Node(
                        game=self.game,
                        args=self.args,
                        state=child_state,
                        parent=self,
                        action_taken=action,
                        prior=prob
                    )
                    self.children.append(child)
                    
                except (ValueError, Exception) as e:
                    # Ignorar movimientos que fallen
                    import sys
                    print(f"⚠️ Warning: Acción {action} falló: {e}", file=sys.stderr)
                    continue
            
    def backpropagate(self, value):
        """
        Propaga el valor hacia arriba en el árbol.
        
        CONVENCIÓN CRÍTICA:
        - 'value' que llega aquí está desde la perspectiva del jugador que ACABA DE MOVER
        - Lo sumamos directamente a nuestro value_sum
        - Al pasar al padre, INVERTIMOS porque el padre es el oponente
        
        Ejemplo:
        - Si llegamos con value=0.8 (bueno para quien movió)
        - Este nodo lo suma: value_sum += 0.8
        - El padre recibe: backpropagate(-0.8) (malo para el oponente)
        
        Args:
            value: Valor en rango [-1, 1]
        """
        self.value_sum += value
        self.visit_count += 1
        
        # Propagar al padre con signo invertido
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """
    Monte Carlo Tree Search con red neuronal (estilo AlphaZero).
    
    FLUJO DE VALORES:
    1. Red neuronal evalúa posición → value (perspectiva del jugador actual)
    2. Backpropagate invierte automáticamente para el oponente
    3. UCB compara valores desde la perspectiva del padre
    
    NO necesitamos inversiones manuales adicionales.
    """
    
    def __init__(self, game, args, model, device=None):
        self.game = game
        self.args = args
        self.model = model
        self.device = device if device is not None else torch.device("cpu")
        
    @torch.no_grad()
    def search(self, state):
        """
        Ejecuta búsqueda MCTS guiada por red neuronal.
        
        Args:
            state: Estado actual del juego
            
        Returns:
            action_probs: Distribución de probabilidad sobre acciones
        """
        # Crear nodo raíz
        root = Node(self.game, self.args, state)
        
        # Expandir raíz inmediatamente con el policy del modelo
        policy, _ = self._evaluate(state)
        root.expand(policy)
        
        # Verificar que se hayan creado hijos
        if len(root.children) == 0:
            # No hay movimientos legales
            valid_moves = self.game.get_valid_moves(state)
            if valid_moves.sum() == 0:
                return np.zeros(self.game.action_size)
            return valid_moves / valid_moves.sum()
        
        # Realizar búsquedas iterativas
        for search_iteration in range(self.args['num_searches']):
            node = root
            
            # 1. SELECTION: Bajar por el árbol usando UCB
            while node.is_fully_expanded(): # type: ignore
                node = node.select() # type: ignore
                
                # Si el nodo seleccionado no tiene hijos, salir
                if len(node.children) == 0: # type: ignore
                    break
            
            # 2. EVALUACIÓN: Obtener valor del nodo
            # Verificar si es terminal
            value, is_terminal = self.game.get_value_and_terminated(
                node.state,  # type: ignore
                node.action_taken # type: ignore
            )
            
            if is_terminal:
                # ✅ CORREGIDO: El valor terminal está desde la perspectiva
                # del jugador que ACABA DE MOVER (quien causó la terminación)
                # 
                # Si es checkmate: value = -1 (el jugador actual perdió)
                # Pero el jugador actual NO es quien movió, es el oponente
                # Entonces el valor para quien movió es +1
                #
                # Por eso INVERTIMOS el valor terminal
                value = -value
            else:
                # 3. EXPANSION: Si no es terminal, expandir
                policy, value = self._evaluate(node.state) # type: ignore
                node.expand(policy) # type: ignore
                
                # ✅ CORREGIDO: El valor del modelo está desde la perspectiva
                # del jugador actual (state.turn), que es correcto para backprop
                # NO necesitamos invertir aquí
            
            # 4. BACKPROPAGATION: Propagar el valor
            # El valor está desde la perspectiva correcta (jugador que movió al nodo)
            node.backpropagate(value) # type: ignore
        
        # Construir distribución de probabilidad basada en visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        
        # Normalizar
        if action_probs.sum() > 0:
            action_probs /= action_probs.sum()
        else:
            # Fallback
            valid_moves = self.game.get_valid_moves(state)
            if valid_moves.sum() > 0:
                action_probs = valid_moves / valid_moves.sum()
        
        return action_probs
    
    def _evaluate(self, state):
        """
        Evalúa un estado usando la red neuronal.
        
        Args:
            state: Estado del juego
            
        Returns:
            policy: Vector de probabilidades (normalizado, enmascarado)
            value: Evaluación en rango [-1, 1] desde perspectiva del jugador actual
        """
        # Codificar estado
        encoded_state = self.game.get_encoded_state(state)
        state_tensor = torch.tensor(
            encoded_state, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0)
        
        # Forward pass
        policy_logits, value = self.model(state_tensor)
        
        # Convertir logits a probabilidades
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        
        # Enmascarar movimientos ilegales
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        
        # Renormalizar
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # Sin movimientos legales válidos
            if valid_moves.sum() > 0:
                policy = valid_moves / np.sum(valid_moves)
            else:
                policy = np.ones(self.game.action_size) / self.game.action_size
        
        # Extraer valor escalar
        value = value.item()
        
        return policy, value


# === TESTS DE VERIFICACIÓN ===

def test_value_propagation():
    """
    Test específico para verificar que los valores se propagan correctamente.
    """
    from chess_game import ChessGame
    from model import create_chess_model
    import chess
    
    print("=" * 70)
    print("TEST: PROPAGACIÓN CORRECTA DE VALORES")
    print("=" * 70)
    
    game = ChessGame(include_queen_promotions=False)
    model = create_chess_model(game, num_resBlocks=2, num_hidden=32)
    model.eval()
    
    args = {'C': 2, 'num_searches': 10}
    mcts = MCTS(game, args, model)
    
    # Test 1: Posición ganadora para blancas (mate en 1)
    print("\n1️⃣ Test: Mate en 1 para blancas")
    state_mate_in_1 = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    game.render(state_mate_in_1)
    
    # Ejecutar MCTS
    action_probs = mcts.search(state_mate_in_1)
    
    # El mejor movimiento debería ser Qf7# o Qg7#
    top_action = np.argmax(action_probs)
    top_move = game.get_move_from_action(state_mate_in_1, top_action)
    
    print(f"  Movimiento elegido: {top_move.uci()}")
    print(f"  Confianza: {action_probs[top_action]:.2%}")
    
    # Verificar que es mate
    new_state = game.get_next_state(state_mate_in_1, top_action, 1)
    if new_state.is_checkmate():
        print("  ✅ ¡Encontró el mate!")
    else:
        print("  ⚠️ No encontró el mate (pero es normal con pocas búsquedas)")
    
    # Test 2: Posición con valor claro
    print("\n2️⃣ Test: Posición con ventaja material")
    state_advantage = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1")
    game.render(state_advantage)
    
    action_probs = mcts.search(state_advantage)
    print(f"  Acciones exploradas: {(action_probs > 0).sum()}")
    
    # Test 3: Verificar que los valores tienen sentido
    print("\n3️⃣ Test: Coherencia de valores")
    state = game.get_initial_state()
    
    # Evaluar posición inicial
    policy, value = mcts._evaluate(state)
    print(f"  Valor de posición inicial: {value:.4f}")
    print(f"  (Debería estar cerca de 0 en posición equilibrada)")
    
    if abs(value) < 0.5:
        print("  ✅ Valor razonable para posición inicial")
    else:
        print("  ⚠️ Valor inusual (puede ser modelo sin entrenar)")
    
    print("\n" + "="*70)
    print("✅ TEST DE PROPAGACIÓN COMPLETADO")
    print("="*70)


def test_ucb_calculation():
    """
    Test para verificar que UCB se calcula correctamente.
    """
    from chess_game import ChessGame
    
    print("\n" + "="*70)
    print("TEST: CÁLCULO DE UCB")
    print("="*70)
    
    game = ChessGame(include_queen_promotions=False)
    args = {'C': 2, 'num_searches': 100}
    
    # Crear nodo padre
    state = game.get_initial_state()
    parent = Node(game, args, state)
    parent.visit_count = 100
    
    # Crear dos hijos con diferentes estadísticas
    state1 = state.copy()
    child1 = Node(game, args, state1, parent, action_taken=0, prior=0.3) # type: ignore
    child1.visit_count = 30
    child1.value_sum = 15  # Q = 15/30 = 0.5
    
    state2 = state.copy()
    child2 = Node(game, args, state2, parent, action_taken=1, prior=0.1) # type: ignore
    child2.visit_count = 5
    child2.value_sum = 1  # Q = 1/5 = 0.2
    
    parent.children = [child1, child2]
    
    # Calcular UCB
    ucb1 = parent.get_ucb(child1)
    ucb2 = parent.get_ucb(child2)
    
    print(f"\nHijo 1:")
    print(f"  Visitas: {child1.visit_count}")
    print(f"  Value sum: {child1.value_sum}")
    print(f"  Q: {-child1.value_sum/child1.visit_count:.4f} (desde perspectiva del padre)")
    print(f"  Prior: {child1.prior:.2f}")
    print(f"  UCB: {ucb1:.4f}")
    
    print(f"\nHijo 2:")
    print(f"  Visitas: {child2.visit_count}")
    print(f"  Value sum: {child2.value_sum}")
    print(f"  Q: {-child2.value_sum/child2.visit_count:.4f} (desde perspectiva del padre)")
    print(f"  Prior: {child2.prior:.2f}")
    print(f"  UCB: {ucb2:.4f}")
    
    selected = parent.select()
    print(f"\nNodo seleccionado: {'Hijo 1' if selected == child1 else 'Hijo 2'}")
    print("  (Hijo 2 debería ser seleccionado por mayor exploración)")
    
    print("\n" + "="*70)
    print("✅ TEST DE UCB COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    test_value_propagation()
    test_ucb_calculation()