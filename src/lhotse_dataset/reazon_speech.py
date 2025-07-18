from pathlib import Path
from typing import Generator

import lhotse
from datasets import DatasetDict, load_dataset

from lhotse_dataset.base import BaseCorpus, Language


# NOTE: librosaの依存がある？ので別途インストール必要かも
class ReazonSpeech(BaseCorpus):
    dataset_size: str

    def __init__(self, dataset_size: str = "all"):
        """dataset_size: tiny, small, medium, large, all"""
        self.dataset_size = dataset_size

    @property
    def url(self) -> str:
        return "https://huggingface.co/datasets/reazon-research/reazonspeech"

    @property
    def language(self) -> Language:
        return Language.JA

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        ds = load_dataset(
            "reazon-research/reazonspeech", self.dataset_size, trust_remote_code=True
        )
        assert type(ds) is DatasetDict

        for sample in ds["train"]:
            recording = lhotse.Recording.from_file(sample["audio"]["path"])  # type: ignore

            audio_id = Path(sample["name"]).stem  # type: ignore

            supervision = lhotse.SupervisionSegment(
                id=f"segment_{audio_id}",
                recording_id=recording.id,
                start=0,
                duration=recording.duration,
                channel=0,
                text=sample["transcription"],  # type: ignore
                language=self.language.value,
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
