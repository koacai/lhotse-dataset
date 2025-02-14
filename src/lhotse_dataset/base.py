import enum
from abc import ABCMeta, abstractmethod
from typing import Generator

import lhotse


@enum.unique
class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"


@enum.unique
class Language(enum.Enum):
    JA = "ja"


class BaseCorpus(metaclass=ABCMeta):
    @property
    @abstractmethod
    def url(self) -> str:
        pass

    @property
    @abstractmethod
    def language(self) -> Language:
        pass

    @abstractmethod
    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        pass
