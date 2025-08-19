import os
import torch
import torch.distributed as dist

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))

    try:
        dist.init_process_group(
            backend='nccl',
            rank=int(os.environ['RANK']),
            world_size=int(os.environ['WORLD_SIZE'])
        )
        print("NCCL backend initialized successfully")
        
        device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}')
        x = torch.tensor([1.0]).to(device)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        print(f"all_reduce test passed, result: {x.item()}")

        dist.destroy_process_group()
    except Exception as e:
        print(f"NCCL initialization failed: {e}")

if __name__ == '__main__':
    main()