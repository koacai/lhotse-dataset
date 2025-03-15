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
                tmp_path = Path(tmp_dir) / f"{dataset_type}.tar.gz"
                download_file(download_url, tmp_path)

                with tarfile.open(tmp_path) as tar:
                    trans_tarinfos = [
                        member
                        for member in tar.getmembers()
                        if member.name.endswith(".trans.txt")
                    ]

                    speakers: dict[str, SpeakerInfo] = {}

                    speakers_file = tar.extractfile("LibriSpeech/SPEAKERS.TXT")
                    assert speakers_file is not None
                    lines = speakers_file.read().decode().strip().split("\n")
                    for line in lines:
                        if line.startswith(";"):
                            continue
                        speaker_id = line.split("|")[0].strip()
                        gender_str = line.split("|")[1].strip()
                        name = "".join(line.split("|")[4:]).strip()
                        if gender_str == "M":
                            gender = Gender.MALE
                        elif gender_str == "F":
                            gender = Gender.FEMALE
                        else:
                            raise ValueError(f"invalid gender str: {gender_str}")
                        speakers[speaker_id] = SpeakerInfo(
                            id=speaker_id, name=name, gender=gender
                        )

                    for trans_tarinfo in trans_tarinfos:
                        trans_file = tar.extractfile(trans_tarinfo)
                        assert trans_file is not None
                        lines = trans_file.read().decode().strip().split("\n")

                        for line in lines:
                            stem = line.split(" ")[0]
                            speaker_id = stem.split("-")[0]
                            transcript = " ".join(line.split(" ")[1:])
                            audio_id = f"librispeech_{dataset_type}_{stem}"

                            wav_path = Path(trans_tarinfo.name).parent / f"{stem}.flac"

                            audio_file = tar.extractfile(str(wav_path))
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
                                id=f"{audio_id}",
                                start=0,
                                duration=recording.duration,
                                channel=0,
                                supervisions=[supervision],
                                recording=recording,
                            )
                            yield cut
