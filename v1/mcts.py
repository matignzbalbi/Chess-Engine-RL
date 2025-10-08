import numpy as np
import math

## Definimos MCTS

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.expandable_moves = game.get_valid_moves(state)
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        # Fórmula UCB1: Q(s,a) + C * sqrt(log(N(s)) / N(s,a))
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        # Seleccionar una acción no expandida al azar
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        # Crear nuevo estado aplicando la acción
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)
        
        # Crear nodo hijo
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        """Realiza un rollout aleatorio desde este nodo hasta un estado terminal"""
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        # Rollout: juego aleatorio hasta el final
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value    
            
            rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        """Propaga el resultado hacia arriba en el árbol"""
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        
    def search(self, state):
        """
        Ejecuta búsqueda MCTS desde el estado dado.
        
        Args:
            state: Estado actual del juego
            
        Returns:
            action_probs: Vector de probabilidades para cada acción
        """
        root = Node(self.game, self.args, state)
        
        # Realizar búsquedas iterativas
        for search in range(self.args['num_searches']):
            node = root
            
            # 1. SELECTION: Bajar por el árbol usando UCB
            while node.is_fully_expanded(): # pyright: ignore[reportOptionalMemberAccess]
                node = node.select() # type: ignore
            
            # Verificar si llegamos a un nodo terminal
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken) # type: ignore
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                # 2. EXPANSION: Expandir un hijo
                node = node.expand() # type: ignore
                # 3. SIMULATION: Rollout aleatorio
                value = node.simulate()
            
            # 4. BACKPROPAGATION: Propagar resultado
            node.backpropagate(value)     # type: ignore
        
        # Retornar distribución de probabilidad basada en visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs