import torch
import torch.distributed as dist
from megatron_het.p2p_backend import init_p2p_comm, _HET_COMM

def main():
    dist.init_process_group(backend="gloo")

    init_p2p_comm()
    default_group = torch.distributed.group.WORLD
    comm = _HET_COMM[default_group.group_name]

    rank = dist.get_rank()
    if rank == 0:
        x = torch.arange(10, dtype=torch.float32, device="cuda")
        print(f"[Rank 0] sending {x}")
        comm.send(x, dst=1)
        print("[Rank 0] send done")
    elif rank == 1:
        y = torch.empty(10, dtype=torch.float32, device="cuda")
        comm.recv(y, src=0)
        print(f"[Rank 1] received {y}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=2 test_p2p_backend_tcp.py
