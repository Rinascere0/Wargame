from abc import ABC, abstractmethod
from typing import List


class BaseAgent(ABC):
    @abstractmethod
    def setup(self, setup_info: dict) -> None:
        pass

    def command(self, observation: dict) -> List[dict]:
        raise NotImplementedError
    
    @abstractmethod
    def deploy(self, observation: dict) -> List[dict]:
        pass

    @abstractmethod
    def step(self, observation: dict) -> List[dict]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
