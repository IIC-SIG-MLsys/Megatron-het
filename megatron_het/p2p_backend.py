from abc import ABC, abstractmethod
from typing import List, Any, Optional
import torch
from enum import Enum

class P2pBackend(Enum):
    HMCComm = 1
    TCPComm = 2

class Request:
    """
    A handle for asynchronous communication operations.
    This object can be subclassed to hold backend-specific resources
    (e.g., MPI request handles, CUDA events, sockets, etc.).
    """
    def __init__(self):
        self._completed = False

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._wait_impl(timeout)

    @abstractmethod
    def _wait_impl(self, timeout: Optional[float]) -> bool:
        pass

    def is_completed(self) -> bool:
        return self._completed


class CommBase(ABC):
    """
    Abstract base class for communication backends.
    Defines a common interface for asynchronous send/recv and batch operations.
    """
    @abstractmethod
    def isend(self, tensor: torch.Tensor, dst: int) -> Request:
        pass

    @abstractmethod
    def irecv(self, tensor: torch.Tensor, src: int) -> Request:
        pass

    @abstractmethod
    def batch_isend_irecv(self, ops: List[Request]) -> List[Request]:
        pass

    @abstractmethod
    def get_rank(self) -> int:
        pass

    def send(self, tensor: torch.Tensor, dst: int):
        req = self.isend(tensor, dst)
        req.wait()

    def recv(self, tensor: torch.Tensor, src: int):
        req = self.irecv(tensor, src)
        req.wait()

#####################################################
# HMC Communicator
#####################################################
class HMCRequest(Request):
    def _wait_impl(self, timeout):
        return super()._wait_impl(timeout)
    
class HMCComm(CommBase):
    def __init__(self, rank: int, rank_ip: dict = {}):
        """
        rank_ip: { rank : (192.168.1.2, 12345) }
        """
        super().__init__()
        self.rank = rank
        self.rank_ip=rank_ip

    def isend(self, tensor: torch.Tensor, dst: int):
        return super().isend(tensor, dst)
    
    def irecv(self, tensor: torch.Tensor, src: int):
        return super().irecv(tensor, src)
    
    def batch_isend_irecv(self, ops):
        return super().batch_isend_irecv(ops)
    
#####################################################
# TCP Communicator
#####################################################
import socket
import threading
import struct

class TCPRequest(Request):
    def __init__(self, thread: threading.Thread):
        super().__init__()
        self._thread = thread
        thread.start()

    def _wait_impl(self, timeout: Optional[float]) -> bool:
        self._thread.join(timeout)
        if not self._thread.is_alive():
            self._completed = True
        return self._completed


class TCPComm(CommBase):
    def __init__(self, rank: int, rank_ip: dict):
        """
        rank_ip: { rank : (ip, port) }
        """
        super().__init__()
        self.rank = rank
        self.rank_ip = rank_ip

        ip, port = self.rank_ip[self.rank]
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((ip, port))
        self.server_sock.listen()

        self.connections = {}
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _listen_loop(self):
        while True:
            conn, addr = self.server_sock.accept()
            # 先收一个 int 表示对方 rank
            peer_rank = struct.unpack("I", conn.recv(4))[0]
            self.connections[peer_rank] = conn

    def _connect(self, dst: int):
        if dst not in self.connections:
            ip, port = self.rank_ip[dst]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            sock.sendall(struct.pack("I", self.rank))
            self.connections[dst] = sock
        return self.connections[dst]

    def isend(self, tensor: torch.Tensor, dst: int) -> Request:
        def send_fn():
            sock = self._connect(dst)
            cpu_tensor = tensor.detach().cpu()
            data = cpu_tensor.numpy().tobytes()

            # send meta: dtype, shape, size
            meta = f"{str(cpu_tensor.dtype)}|{','.join(map(str, cpu_tensor.shape))}".encode()
            sock.sendall(struct.pack("I", len(meta)))
            sock.sendall(meta)

            # send data
            sock.sendall(struct.pack("I", len(data)))
            sock.sendall(data)

        return TCPRequest(threading.Thread(target=send_fn))

    def irecv(self, tensor: torch.Tensor, src: int) -> Request:
        def recv_fn():
            while src not in self.connections:
                pass
            sock = self.connections[src]

            # recv meta
            meta_len = struct.unpack("I", sock.recv(4))[0]
            meta = sock.recv(meta_len).decode()
            dtype_str, shape_str = meta.split("|")
            shape = tuple(map(int, shape_str.split(",")))

            # recv data → use bytearray
            data_len = struct.unpack("I", sock.recv(4))[0]
            buf = bytearray()
            while len(buf) < data_len:
                buf += sock.recv(data_len - len(buf))

            # create tensor from writable buffer
            cpu_tensor = torch.frombuffer(
                buf,
                dtype=getattr(torch, dtype_str.split(".")[-1])
            ).reshape(shape).clone()  # clone to ensure ownership

            # copy to target device
            tensor.copy_(cpu_tensor.to(tensor.device))

        return TCPRequest(threading.Thread(target=recv_fn))

    def batch_isend_irecv(self, ops: List[Request]) -> List[Request]:
        return ops

    def get_rank(self) -> int:
        return self.rank
    

#####################################################
# P2P Init
#####################################################
from typing import Dict, Tuple, List
import torch.distributed as dist

_HET_COMM: Dict[str, 'CommBase'] = {} # gourp_name : HMCComm

import socket

def get_random_local_ip_port():
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    if ip.startswith("127."):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    
    return ip, port

def get_rank_ip_port_map(group: dist.ProcessGroup) -> Dict[int, Tuple[str, int]]:
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized.")

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    # rank n's ip, port
    ip, port = get_random_local_ip_port()
    local_info = (rank, ip, port)

    # group all_gather
    object_list = [None for _ in range(world_size)]
    dist.all_gather_object(object_list, local_info, group=group)

    rank_ip_map = {r: (ip, port) for r, ip, port in object_list}
    return rank_ip_map

def init_p2p_comm(group: torch.distributed.ProcessGroup = None, p2p_backend: P2pBackend = P2pBackend.TCPComm):
    global _HET_COMM
    if group == None:
        group = torch.distributed.group.WORLD
    if group.group_name in _HET_COMM.keys():
        return
    if p2p_backend == P2pBackend.HMCComm:
        # print(f"Init p2p HMCComm for group {group.group_name}")
        _HET_COMM[group.group_name] = HMCComm(rank=dist.get_rank(), rank_ip=get_rank_ip_port_map(group))
    elif p2p_backend == P2pBackend.TCPComm:
        # print(f"Init p2p TCPComm for group {group.group_name}")
        _HET_COMM[group.group_name] = TCPComm(rank=dist.get_rank(), rank_ip=get_rank_ip_port_map(group))
    else:
        raise ValueError("p2p_backend must be set")
        