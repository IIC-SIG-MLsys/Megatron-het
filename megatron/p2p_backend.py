from abc import ABC, abstractmethod
from typing import List, Any, Optional


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
    def isend(self, tensor: Any, dst: int) -> Request:
        pass

    @abstractmethod
    def irecv(self, tensor: Any, src: int) -> Request:
        pass

    @abstractmethod
    def batch_isend_irecv(self, ops: List[Request]) -> List[Request]:
        pass

    @abstractmethod
    def get_rank(self) -> int:
        pass

    def send(self, tensor: Any, dst: int):
        req = self.isend(tensor, dst)
        req.wait()

    def recv(self, tensor: Any, src: int):
        req = self.irecv(tensor, src)
        req.wait()


class HMCRequest(Request):
    def _wait_impl(self, timeout):
        return super()._wait_impl(timeout)
    
class HMCComm(CommBase):
    ip = None # "127.0.0.1"
    port = None # "12345"
    rank_ip_port = {} # { rank : (192.168.1.2, 12345) }

    def __init__(self, ip: str = "127.0.0.1", port: int = "12345", rank_ip: dict = {}):
        """
        rank_ip: { rank : (192.168.1.2, 12345) }
        """
        super().__init__()
        self.ip = ip
        self.port = port
        self.rank_ip=rank_ip

    def isend(self, tensor, dst):
        return super().isend(tensor, dst)
    
    def irecv(self, tensor, src):
        return super().irecv(tensor, src)
    
    def batch_isend_irecv(self, ops):
        return super().batch_isend_irecv(ops)
    