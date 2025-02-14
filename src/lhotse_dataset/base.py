import enum
from abc import ABCMeta, abstractmethod
from typing import Generator

import lhotse


class BaseCorpus(metaclass=ABCMeta):
    @property
    @abstractmethod
    def url(self) -> str:
        pass

    @abstractmethod
    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        pass


@enum.unique
class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"


@enum.unique
class Language(enum.Enum):
    JA = "ja"
