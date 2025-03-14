import enum
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generator

import lhotse


@enum.unique
class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"


@enum.unique
class Language(enum.Enum):
    JA = "ja"
    EN = "en"


@dataclass
class SpeakerInfo:
    id: str
    name: str
    gender: Gender


class BaseCorpus(metaclass=ABCMeta):
    @abstractmethod
    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        pass
