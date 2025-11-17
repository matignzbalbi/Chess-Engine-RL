
import logging
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(backend='ccl'):
    """
    Configura entorno distribuido para Intel GPUs.
    
    Args:
        backend: 'ccl' para Intel GPUs, 'nccl' para NVIDIA, 'gloo' para CPU
    
    Returns:
        rank, world_size, local_rank
    """
    
    # Verificar variables de entorno
    if 'RANK' not in os.environ:
        logging.warning("Variables de entorno de distributed no encontradas")
        logging.warning("Ejecutando en modo single-GPU")
        return None, 1, None
    
    # Obtener info del proceso
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Inicializar proceso distribuido
    if not dist.is_initialized():
        # Para Intel GPUs, usar CCL backend
        if backend == 'ccl':
            try:
                import intel_extension_for_pytorch as ipex # type: ignore
                dist.init_process_group(
                    backend='ccl',
                    init_method='env://',
                    rank=rank,
                    world_size=world_size
                )
                logging.info(f"✓ Distributed setup: rank {rank}/{world_size} (CCL backend)")
            except Exception as e:
                logging.error(f"Error inicializando CCL: {e}")
                logging.info("Falling back to gloo backend")
                dist.init_process_group(
                    backend='gloo',
                    init_method='env://',
                    rank=rank,
                    world_size=world_size
                )
        else:
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                rank=rank,
                world_size=world_size
            )
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Limpia proceso distribuido"""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model, device, device_type='xpu', local_rank=None):

    
    if not dist.is_initialized():
        logging.warning("Distributed no inicializado. Retornando modelo sin DDP")
        return model
    
    # Mover modelo al dispositivo correcto
    if device_type == 'xpu' and local_rank is not None:
        device = torch.device(f'xpu:{local_rank}')
    
    model = model.to(device)
    
    # Envolver con DDP
    model = DDP(
        model,
        device_ids=[local_rank] if device_type in ['xpu', 'cuda'] else None,
        output_device=local_rank if device_type in ['xpu', 'cuda'] else None
    )
    
    logging.info(f"✓ Modelo envuelto con DistributedDataParallel")
    
    return model


def is_main_process():
    """Retorna True si es el proceso principal (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Obtiene el rank del proceso actual"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Obtiene el número total de procesos"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Sincroniza todos los procesos"""
    if dist.is_initialized():
        dist.barrier()


def reduce_value(value, op='mean'):
    """
    Reduce un valor entre todos los procesos.
    
    Args:
        value: Valor a reducir (float o tensor)
        op: 'mean', 'sum', 'max', 'min'
    
    Returns:
        Valor reducido
    """
    
    if not dist.is_initialized():
        return value
    
    # Convertir a tensor si es necesario
    if not isinstance(value, torch.Tensor):
        value_tensor = torch.tensor(value, dtype=torch.float32)
    else:
        value_tensor = value.clone()
    
    # Reducir
    if op == 'mean':
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        value_tensor /= get_world_size()
    elif op == 'sum':
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    elif op == 'max':
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
    elif op == 'min':
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MIN)
    
    return value_tensor.item() if not isinstance(value, torch.Tensor) else value_tensor


# Script de ejemplo para lanzar entrenamiento distribuido
def create_launch_script():
    """Crea script de ejemplo para lanzar con torchrun"""
    
    script = """#!/bin/bash
# launch_distributed.sh
# Script para lanzar entrenamiento distribuido en Intel GPUs

# Número de GPUs Intel a usar
NUM_GPUS=2

# Ejecutar con torchrun (recomendado)
torchrun \\
    --nproc_per_node=$NUM_GPUS \\
    --nnodes=1 \\
    --node_rank=0 \\
    test_entrenamiento_ddp.py

# Alternativa con python -m torch.distributed.launch (deprecated pero funciona)
# python -m torch.distributed.launch \\
#     --nproc_per_node=$NUM_GPUS \\
#     --nnodes=1 \\
#     --node_rank=0 \\
#     test_entrenamiento_ddp.py
"""
    
    with open('launch_distributed.sh', 'w') as f:
        f.write(script)
    
    os.chmod('launch_distributed.sh', 0o755)
    logging.info("✓ Script de lanzamiento creado: launch_distributed.sh")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n" + "="*70)
    print("TEST DE CONFIGURACIÓN DISTRIBUIDA")
    print("="*70 + "\n")
    
    # Intentar setup distribuido
    rank, world_size, local_rank = setup_distributed(backend='ccl')
    
    if rank is not None:
        print(f"Proceso {rank}/{world_size} inicializado")
        print(f"Local rank: {local_rank}")
        
        # Test de comunicación
        if world_size > 1:
            test_tensor = torch.tensor([rank], dtype=torch.float32)
            print(f"Rank {rank}: tensor antes de reduce = {test_tensor.item()}")
            
            reduced = reduce_value(test_tensor, op='sum')
            print(f"Rank {rank}: tensor después de reduce = {reduced}")
            
            barrier()
            
            if is_main_process():
                print("\n✓ Test de comunicación exitoso")
        
        cleanup_distributed()
    else:
        print("No se detectó entorno distribuido")
        print("Para lanzar en modo distribuido, usá:")
        print("  torchrun --nproc_per_node=2 script.py")
    
    # Crear script de lanzamiento
    create_launch_script()
    
    print("\n" + "="*70)
    print("✅ CONFIGURACIÓN COMPLETADA")
    print("="*70)