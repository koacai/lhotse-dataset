import io
from typing import Generator

import soundfile as sf
from datasets import load_dataset
from lhotse import MonoCut, Recording

from lhotse_dataset.base import BaseCorpus, Language


class SeamlessInteraction(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://ai.meta.com/research/seamless-interaction/"

    @property
    def language(self) -> Language:
        return Language.EN

    @property
    def shard_size(self) -> int:
        return 1000

    @property
    def labels(self) -> list[str]:
        return ["improvised", "naturalistic"]

    @property
    def splits(self) -> list[str]:
        return ["dev", "test", "train"]

    def get_cuts(self) -> Generator[MonoCut, None, None]:
        for label in self.labels:
            for split in self.splits:
                dataset = load_dataset(
                    "facebook/seamless-interaction", label, split=split, streaming=True
                )
                for item in dataset:
                    assert isinstance(item, dict)

                    id = item["json"]["id"]
                    vendor, session, interaction, participant = id.split("_")

                    buf = io.BytesIO()
                    wav = item["wav"]["array"]
                    sr = item["wav"]["sampling_rate"]
                    sf.write(buf, wav, sr, format="WAV")

                    recording = Recording.from_bytes(
                        buf.getvalue(), recording_id=f"recording_{id}"
                    )

                    cut = MonoCut(
                        id=id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        supervisions=[],
                        recording=recording,
                        custom={
                            "vendor": vendor,
                            "session": session,
                            "interaction": interaction,
                            "participant": participant,
                        },
                    )

                    yield cut
