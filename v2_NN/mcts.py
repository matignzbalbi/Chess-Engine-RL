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
        self.value_sum = 0  # Suma de valores desde la perspectiva del jugador actual
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
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
     
        # Obtener máscara de movimientos legales
        valid_moves = self.game.get_valid_moves(self.state)
        
        for action, prob in enumerate(policy):
            # Solo expandir acciones con probabilidad > 0 Y legales
            if prob > 0 and valid_moves[action] > 0:
                try:
                    # Crear nuevo estado aplicando el movimiento
                    child_state = self.state.copy()
                    child_state = self.game.get_next_state(child_state, action, 1)
                
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
                    print(f"Acción {action} falló: {e}", file=sys.stderr)
                    continue
            
    def backpropagate(self, value):
    
        self.value_sum += value
        self.visit_count += 1
        
        # Propagar al padre con signo invertido
        if self.parent is not None:
            self.parent.backpropagate(-value)

class MCTS:
    
    def __init__(self, game, args, model, device=None):
        self.game = game
        self.args = args
        self.model = model
        self.device = device if device is not None else torch.device("cpu")
        
    @torch.no_grad()
    def search(self, state):
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
                
                if len(node.children) == 0: # type: ignore
                    break
            
            # 2. EVALUACIÓN: Obtener valor del nodo
            value, is_terminal = self.game.get_value_and_terminated(
                node.state,  # type: ignore
                node.action_taken # type: ignore
            )
            
            if is_terminal:
    
                value = -value
            else:
                # 3. EXPANSION: Si no es terminal, expandir
                policy, value = self._evaluate(node.state) # type: ignore
                node.expand(policy) # type: ignore
  
            # 4. BACKPROPAGATION
            node.backpropagate(value) # type: ignore
        
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        
        if action_probs.sum() > 0:
            action_probs /= action_probs.sum()
        else:
            # Fallback
            valid_moves = self.game.get_valid_moves(state)
            if valid_moves.sum() > 0:
                action_probs = valid_moves / valid_moves.sum()
        
        return action_probs
    
    def _evaluate(self, state):

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


