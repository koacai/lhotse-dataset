import enum
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from lhotse.cut import Cut
from lhotse.shar import SharWriter
from tqdm import tqdm


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
    def get_cuts(self) -> Generator[Cut, None, None]:
        pass

    @property
    def shard_size(self) -> int:
        return 1000

    def write_shar(self, output_dir: Path, shard_size: int | None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        if shard_size is None:
            shard_size = self.shard_size

        with SharWriter(
            str(output_dir), fields={"recording": "flac"}, shard_size=shard_size
        ) as writer:
            for cut in tqdm(self.get_cuts()):
                writer.write(cut)
