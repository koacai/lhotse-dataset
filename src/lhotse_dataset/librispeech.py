import tarfile
import tempfile
from pathlib import Path
from typing import Generator

import lhotse

from lhotse_dataset.base import BaseCorpus, Gender, Language, SpeakerInfo
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
        }

    @property
    def language(self) -> Language:
        return Language.EN

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for dataset_type, download_url in self.download_url.items():
                tmp_dir_path = Path(tmp_dir)
                tmp_path = tmp_dir_path / f"{dataset_type}.tar.gz"
                download_file(download_url, tmp_path)

                with tarfile.open(tmp_path) as tar:
                    tar.extractall(tmp_dir_path, filter="fully_trusted")

                speakers = {}
                speakers_file_path = Path(tmp_dir) / "LibriSpeech/SPEAKERS.TXT"
                with open(speakers_file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith(";"):
                            continue
                        parts = [p.strip() for p in line.split("|")]
                        speaker_id, gender_str, name = (
                            parts[0],
                            parts[1],
                            "".join(parts[4:]),
                        )
                        gender = Gender.MALE if gender_str == "M" else Gender.FEMALE
                        speakers[speaker_id] = SpeakerInfo(
                            id=speaker_id, name=name, gender=gender
                        )

                trans_files = list(tmp_dir_path.glob("LibriSpeech/**/*.trans.txt"))
                for trans_file_path in trans_files:
                    with open(trans_file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split(" ")
                        stem, transcript = parts[0], " ".join(parts[1:])
                        speaker_id = stem.split("-")[0]
                        audio_id = f"librispeech_{dataset_type}_{stem}"

                        wav_file_path = trans_file_path.parent / f"{stem}.flac"

                        recording = lhotse.Recording.from_file(str(wav_file_path))

                        supervision = lhotse.SupervisionSegment(
                            id=f"segment_{audio_id}",
                            recording_id=recording.id,
                            start=0,
                            duration=recording.duration,
                            channel=0,
                            text=transcript,
                            language=self.language.value,
                            speaker=speaker_id,
                            gender=speakers[speaker_id].gender.value,
                            custom={
                                "dataset_type": dataset_type,
                                "speaker_name": speakers[speaker_id].name,
                            },
                        )

                        cut = lhotse.MonoCut(
                            id=audio_id,
                            start=0,
                            duration=recording.duration,
                            channel=0,
                            supervisions=[supervision],
                            recording=recording,
                        )
                        yield cut
