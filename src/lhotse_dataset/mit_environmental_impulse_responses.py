import uuid
from typing import Generator

from datasets import DatasetDict, load_dataset
from lhotse import MonoCut, Recording

from lhotse_dataset.base import BaseCorpus


class MITEnvironmentalImpulseResponses(BaseCorpus):
    @property
    def shard_size(self) -> int:
        return 10

    def get_cuts(self) -> Generator[MonoCut, None, None]:
        ds = load_dataset("davidscripka/MIT_environmental_impulse_responses")
        assert isinstance(ds, DatasetDict)

        for data in ds["train"]:
            assert isinstance(data, dict)
            audio = data["audio"]

            id = uuid.uuid4().hex
            recording = Recording.from_file(
                audio["path"], recording_id=f"recording_{id}"
            )

            cut = MonoCut(
                id=id,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )

            yield cut
