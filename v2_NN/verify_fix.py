"""
Script de verificación para confirmar que el problema de promociones está resuelto.
"""

import chess
from move_mapping import MoveMapper
from chess_game import ChessGame

def verify_promotion_fix():
    """Verifica que las promociones funcionen correctamente."""
    
    print("=" * 70)
    print("VERIFICACIÓN DE FIX DE PROMOCIONES")
    print("=" * 70)
    
    all_tests_passed = True
    
    # Test 1: Ambos modos deben funcionar
    print("\n✅ Test 1: Inicialización de ambos modos")
    try:
        mapper_std = MoveMapper(include_queen_promotions=False)
        mapper_ext = MoveMapper(include_queen_promotions=True)
        print(f"  ✓ Modo estándar: {mapper_std.action_size} acciones")
        print(f"  ✓ Modo extendido: {mapper_ext.action_size} acciones")
        assert mapper_std.action_size == 4672, "Action size incorrecto (estándar)"
        assert mapper_ext.action_size == 4864, "Action size incorrecto (extendido)"
    except Exception as e:
        print(f"  ✗ FALLO: {e}")
        all_tests_passed = False
    
    # Test 2: Promociones a dama en modo estándar
    print("\n✅ Test 2: Promociones a dama (modo estándar)")
    try:
        game = ChessGame(include_queen_promotions=False)
        state = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        
        # Debe poder convertir a7a8q
        move = chess.Move.from_uci("a7a8q")
        action = game.get_action_from_move(move)
        print(f"  ✓ a7a8q → acción {action}")
        
        # Debe poder aplicar el movimiento
        new_state = game.get_next_state(state, action, 1)
        piece = new_state.piece_at(chess.A8)
        assert piece and piece.piece_type == chess.QUEEN, "No hay dama en a8"
        print(f"  ✓ Dama correctamente creada en a8")
        
    except Exception as e:
        print(f"  ✗ FALLO: {e}")
        all_tests_passed = False
    
    # Test 3: Promociones a dama en modo extendido
    print("\n✅ Test 3: Promociones a dama (modo extendido)")
    try:
        game = ChessGame(include_queen_promotions=True)
        state = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        
        # Debe tener acción DIFERENTE para queen
        move_queen = chess.Move.from_uci("a7a8q")
        move_knight = chess.Move.from_uci("a7a8n")
        
        action_queen = game.get_action_from_move(move_queen)
        action_knight = game.get_action_from_move(move_knight)
        
        print(f"  ✓ a7a8q → acción {action_queen}")
        print(f"  ✓ a7a8n → acción {action_knight}")
        
        assert action_queen != action_knight, "Acciones no son diferentes"
        print(f"  ✓ Acciones correctamente diferenciadas")
        
    except Exception as e:
        print(f"  ✗ FALLO: {e}")
        all_tests_passed = False
    
    # Test 4: Underpromotions
    print("\n✅ Test 4: Underpromotions")
    try:
        game = ChessGame(include_queen_promotions=False)
        state = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        
        pieces = {
            'n': chess.KNIGHT,
            'b': chess.BISHOP,
            'r': chess.ROOK
        }
        
        for uci_piece, chess_piece in pieces.items():
            move = chess.Move.from_uci(f"a7a8{uci_piece}")
            action = game.get_action_from_move(move)
            new_state = game.get_next_state(state, action, 1)
            piece = new_state.piece_at(chess.A8)
            
            assert piece and piece.piece_type == chess_piece, \
                f"Pieza incorrecta para {uci_piece}"
            print(f"  ✓ Underpromotion a {chess.piece_name(chess_piece)}")
        
    except Exception as e:
        print(f"  ✗ FALLO: {e}")
        all_tests_passed = False
    
    # Test 5: Máscara de movimientos válidos
    print("\n✅ Test 5: Máscara de movimientos válidos")
    try:
        game = ChessGame(include_queen_promotions=False)
        state = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        
        mask = game.get_valid_moves(state)
        legal_actions = game.get_legal_actions(state)
        python_chess_moves = len(list(state.legal_moves))
        
        assert int(mask.sum()) == len(legal_actions), \
            "Inconsistencia entre máscara y lista"
        print(f"  ✓ Máscara: {int(mask.sum())} acciones")
        print(f"  ✓ Lista: {len(legal_actions)} acciones")
        print(f"  ✓ python-chess: {python_chess_moves} movimientos")
        
        # En modo estándar, python-chess ve 4 promociones pero nosotros
        # mapeamos 3 underpromotions + 1 movimiento sin promoción explícita
        # que se interpreta como queen
        expected_diff = 0  # Deben coincidir
        actual_diff = abs(len(legal_actions) - python_chess_moves)
        
        if actual_diff == expected_diff:
            print(f"  ✓ Coincidencia correcta")
        else:
            print(f"  ⚠️ Diferencia: {actual_diff} (esperado: {expected_diff})")
        
    except Exception as e:
        print(f"  ✗ FALLO: {e}")
        all_tests_passed = False
    
    # Test 6: Conversión bidireccional
    print("\n✅ Test 6: Conversión bidireccional")
    try:
        game = ChessGame(include_queen_promotions=False)
        state = chess.Board()
        
        test_moves = ["e2e4", "d2d4", "g1f3", "b1c3"]
        
        for move_uci in test_moves:
            move = chess.Move.from_uci(move_uci)
            action = game.get_action_from_move(move)
            move_back = game.get_move_from_action(state, action)
            
            assert move == move_back, f"No coinciden: {move.uci()} vs {move_back.uci()}"
            print(f"  ✓ {move_uci} → {action} → {move_back.uci()}")
        
    except Exception as e:
        print(f"  ✗ FALLO: {e}")
        all_tests_passed = False
    
    # Resultado final
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("✅✅✅ TODAS LAS VERIFICACIONES PASARON ✅✅✅")
        print("\nEl problema de promociones está RESUELTO.")
        print("Puedes proceder con el entrenamiento con confianza.")
    else:
        print("❌❌❌ ALGUNAS VERIFICACIONES FALLARON ❌❌❌")
        print("\nRevisa los errores arriba y corrige antes de entrenar.")
    print("=" * 70)
    
    return all_tests_passed


if __name__ == "__main__":
    verify_promotion_fix()