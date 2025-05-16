import tarfile
from pathlib import Path
from typing import Generator

import lhotse

from lhotse_dataset.base import BaseCorpus, Language


class HQYouTube(BaseCorpus):
    tar_path: str

    def __init__(self, tar_path: str) -> None:
        self.tar_path = tar_path

    @property
    def language(self) -> Language:
        return Language.JA

    @property
    def shard_size(self) -> int:
        return 100000

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        with tarfile.TarFile(self.tar_path) as hq_youtube_tar:
            for member in hq_youtube_tar.getmembers():
                if member.isfile():
                    path = Path(member.name)
                    if path.suffix != ".flac":
                        continue
                    audio_file = hq_youtube_tar.extractfile(member)
                    assert audio_file is not None

                    audio_id = path.stem
                    wav_bytes = audio_file.read()
                    recording = lhotse.Recording.from_bytes(
                        wav_bytes, f"recording_{audio_id}"
                    )

                    supervision = lhotse.SupervisionSegment(
                        id=f"segment_{audio_id}",
                        recording_id=recording.id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        language=self.language.value,
                    )
                    cut = lhotse.MonoCut(
                        id=f"{audio_id}",
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        supervisions=[supervision],
                        recording=recording,
                    )
                    yield cut
