from abc import ABCMeta, abstractmethod
from typing import Generator

import lhotse


class BaseCorpus(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def url(self) -> str:
        """Return the URL of the corpus"""

    @abstractmethod
    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        """Return the generator of cuts"""
