import numpy as np
import math
import torch

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior  # P(s,a) - probabilidad del modelo para este movimiento
        
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
        
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
        Calcula UCB mejorado con prior de la red neuronal.
        
        Fórmula AlphaZero: Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Donde:
        - Q(s,a) = valor promedio del nodo hijo
        - P(s,a) = probabilidad prior del modelo
        - N(s) = visitas del nodo padre
        - N(s,a) = visitas del nodo hijo
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            # Normalizar value_sum (de [-1,1] a [0,1] para compatibilidad)
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        
        # Término de exploración basado en el prior
        exploration = self.args['C'] * child.prior * (math.sqrt(self.visit_count) / (child.visit_count + 1))
        
        return q_value + exploration
    
    def expand(self, policy):
        """
        Expande el nodo creando todos los hijos válidos.
        
        CAMBIO CRÍTICO: Ahora verifica que la acción sea legal antes de expandir.
        
        Args:
            policy: Vector de probabilidades del modelo (uno por acción)
        """
        # Obtener máscara de movimientos legales
        valid_moves = self.game.get_valid_moves(self.state)
        
        for action, prob in enumerate(policy):
            # ✅ CAMBIO: Verificar que sea legal Y tenga probabilidad > 0
            if prob > 0 and valid_moves[action] > 0:
                try:
                    # Crear nuevo estado
                    child_state = self.state.copy()
                    child_state = self.game.get_next_state(child_state, action, 1)
                    child_state = self.game.change_perspective(child_state, player=-1)

                    # Crear nodo hijo con el prior del modelo
                    child = Node(self.game, self.args, child_state, self, action, prob)
                    self.children.append(child)
                    
                except (ValueError, Exception) as e:
                    # Si falla get_next_state, ignorar esta acción
                    # Esto puede ocurrir si el mapeo no cubre algún movimiento especial
                    continue
            
    def backpropagate(self, value):
        """
        Propaga el valor hacia arriba en el árbol.
        
        Args:
            value: Valor a propagar (de la red neuronal o del estado terminal)
        """
        self.value_sum += value
        self.visit_count += 1
        
        # Invertir valor para el oponente
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    """
    Monte Carlo Tree Search con red neuronal (estilo AlphaZero).
    
    Diferencias con MCTS clásico:
    - NO hace rollouts aleatorios
    - Usa red neuronal para evaluar posiciones
    - Usa policy de la red para guiar la exploración
    - Expande todos los hijos a la vez (no uno por uno)
    """
    
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
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
        policy, value = self._evaluate(state)
        root.expand(policy)
        
        # Verificar que se hayan creado hijos
        if len(root.children) == 0:
            # No hay movimientos legales - el juego debe haber terminado
            # Retornar distribución uniforme sobre movimientos válidos
            valid_moves = self.game.get_valid_moves(state)
            if valid_moves.sum() == 0:
                # Verdaderamente sin movimientos, retornar vector cero
                return np.zeros(self.game.action_size)
            return valid_moves / valid_moves.sum()
        
        # Realizar búsquedas iterativas
        for search in range(self.args['num_searches']):
            node = root
            
            # 1. SELECTION: Bajar por el árbol usando UCB
            while node.is_fully_expanded(): # type: ignore
                node = node.select() # type: ignore
                
                # Verificar si este nodo tiene hijos
                if len(node.children) == 0: # type: ignore
                    break
            
            # Verificar si llegamos a un nodo terminal
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken) # type: ignore
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                # 2. EXPANSION: Expandir usando la red neuronal
                policy, value = self._evaluate(node.state) # type: ignore
                node.expand(policy) # type: ignore
            
            # 3. BACKPROPAGATION: Propagar el valor
            node.backpropagate(value) # type: ignore    
        
        # Retornar distribución basada en visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        
        # Normalizar
        if action_probs.sum() > 0:
            action_probs /= action_probs.sum()
        else:
            # Fallback: distribución uniforme sobre movimientos válidos
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
            policy: Vector de probabilidades normalizadas para movimientos válidos
            value: Evaluación de la posición (-1 a 1)
        """
        # Codificar el estado
        encoded_state = self.game.get_encoded_state(state)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        
        # Mover a device del modelo si es necesario
        if hasattr(self.model, 'device'):
            state_tensor = state_tensor.to(next(self.model.parameters()).device)
        
        # Forward pass del modelo
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
            # Si todos los movimientos fueron filtrados, distribución uniforme
            if valid_moves.sum() > 0:
                policy = valid_moves / np.sum(valid_moves)
            else:
                # Sin movimientos válidos (posición terminal)
                policy = np.ones(self.game.action_size) / self.game.action_size
        
        # Extraer valor escalar
        value = value.item()
        
        return policy, value