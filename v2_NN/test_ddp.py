import torch
import torch.distributed as dist
import torch.nn as nn
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch  # Critical!

def test_ddp():
    # Inicializar proceso distribuido
    dist.init_process_group(backend='ccl')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Configurar device
    torch.xpu.set_device(local_rank)
    
    print(f"Proceso {local_rank}/{world_size} - GPU: {torch.xpu.current_device()}")
    
    # Modelo simple
    model = nn.Linear(10, 10).xpu()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Test de comunicación
    tensor = torch.ones(1).xpu() * local_rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Proceso {local_rank}: all_reduce result: {tensor.item()}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_ddp()