from pathlib import Path
from typing import Generator

import lhotse

from lhotse_dataset.base import BaseCorpus, Gender, Language


class JIS(BaseCorpus):
    root_dir: Path

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @property
    def language(self) -> Language:
        return Language.JA

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        wav_paths = sorted(list(map(str, self.root_dir.glob("**/*.wav"))))
        for wav_path in wav_paths:
            wav_path = Path(wav_path)
            audio_id = wav_path.stem
            recording = lhotse.Recording.from_file(wav_path, f"recording_{audio_id}")

            speaker, recording_type = wav_path.parent.name.split("_")
            group_name = wav_path.parent.parent.name

            supervision = lhotse.SupervisionSegment(
                id=f"supervision_{audio_id}",
                recording_id=recording.id,
                start=0,
                duration=recording.duration,
                channel=0,
                speaker=speaker,
                language=self.language.value,
                gender=Gender.FEMALE.value,
                custom={"recording_type": recording_type, "group_name": group_name},
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
