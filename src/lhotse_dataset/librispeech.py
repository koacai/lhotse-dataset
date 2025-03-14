import tarfile
import tempfile
from pathlib import Path
from typing import Generator

import lhotse
from tqdm import tqdm

from lhotse_dataset.base import BaseCorpus, Language
from lhotse_dataset.utils import download_file


class LibriSpeech(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://www.openslr.org/12"

    @property
    def download_url(self) -> dict[str, str]:
        return {
            "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
            "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
            "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
            "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
            "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
            "intro-disclaimers": "https://www.openslr.org/resources/12/intro-disclaimers.tar.gz",
            "original-mp3": "https://www.openslr.org/resources/12/original-mp3.tar.gz",
            "original-books": "https://www.openslr.org/resources/12/original-books.tar.gz",
            "raw-metadata": "https://www.openslr.org/resources/12/raw-metadata.tar.gz",
            "md5sum": "https://www.openslr.org/resources/12/md5sum.txt",
        }

    @property
    def language(self) -> Language:
        return Language.EN

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        dataset_types = [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]

        for dataset_type in dataset_types:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / f"{dataset_type}.tar.gz"
                download_file(self.download_url[dataset_type], tmp_path)

                with tarfile.open(tmp_path) as tar:
                    for member in tqdm(tar.getmembers()):
                        if not member.isfile():
                            continue
                        if not member.name.endswith(".flac"):
                            # NOTE: LICENSEが含まれる
                            continue

                        wav_path = Path(member.name)
                        dataset_type = wav_path.parent.parent.parent.name
                        chapter_id, speaker_id, utterance_id = wav_path.stem.split("-")
                        audio_id = f"librispeech_{dataset_type}_{chapter_id}_{speaker_id}_{utterance_id}"  # noqa

                        audio_file = tar.extractfile(member)
                        assert audio_file is not None

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
                            custom={"dataset_type": dataset_type},
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
