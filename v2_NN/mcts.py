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
       
        if child.visit_count == 0:
            q_value = 0
        else:
            # Normalizar value_sum (de [-1,1] a [0,1] para compatibilidad)
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        
        # Término de exploración basado en el prior
        exploration = self.args['C'] * child.prior * (math.sqrt(self.visit_count) / (child.visit_count + 1))
        
        return q_value + exploration
    
    def expand(self, policy):
    
        for action, prob in enumerate(policy):
            if prob > 0:  # Solo expandir movimientos con probabilidad > 0
                # Crear nuevo estado
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                # Crear nodo hijo con el prior del modelo
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
            
    def backpropagate(self, value):
    
        self.value_sum += value
        self.visit_count += 1
        
        # Invertir valor para el oponente
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    
    def __init__(self, game, args, model, device=None):
        self.game = game
        self.args = args
        self.model = model
        # Si no se proporciona device, detectarlo del modelo
        self.device = device or next(model.parameters()).device
        
    @torch.no_grad()
    def search(self, state):
      
        # Crear nodo raíz
        root = Node(self.game, self.args, state)
        
        # Expandir raíz inmediatamente con el policy del modelo
        policy, value = self._evaluate(state)
        root.expand(policy)
        
        # Realizar búsquedas iterativas
        for search in range(self.args['num_searches']):
            node = root
            
            # 1. SELECTION: Bajar por el árbol usando UCB
            while node.is_fully_expanded(): # type: ignore
                node = node.select() # type: ignore
            
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
        action_probs /= np.sum(action_probs)
        return action_probs
    
    def _evaluate(self, state):
     
        # Codificar el estado
        encoded_state = self.game.get_encoded_state(state)
        # ✓ Crear tensor en el device correcto
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
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
            policy = valid_moves / np.sum(valid_moves)
        
        # Extraer valor escalar
        value = value.item()
        
        return policy, value