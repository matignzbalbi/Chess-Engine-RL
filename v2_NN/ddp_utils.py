
import logging
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(backend='ccl'):
   
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
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model, device, device_type='xpu', local_rank=None):

    
    if not dist.is_initialized():
        logging.warning("Distributed no inicializado. Retornando modelo sin DDP")
        return model
    
    if device_type == 'xpu' and local_rank is not None:
        device = torch.device(f'xpu:{local_rank}')
    
    model = model.to(device)
    
    model = DDP(
        model,
        device_ids=[local_rank] if device_type in ['xpu', 'cuda'] else None,
        output_device=local_rank if device_type in ['xpu', 'cuda'] else None
    )
    
    logging.info(f"✓ Modelo envuelto con DistributedDataParallel")
    
    return model


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def reduce_value(value, op='mean'):

    
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



