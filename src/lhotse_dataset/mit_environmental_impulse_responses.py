import io
import uuid
from typing import Generator

import soundfile as sf
from datasets import DatasetDict, load_dataset
from lhotse import MonoCut, Recording
from torchcodec.decoders import AudioDecoder, AudioStreamMetadata

from lhotse_dataset.base import BaseCorpus


class MITEnvironmentalImpulseResponses(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses"

    @property
    def shard_size(self) -> int:
        return 10

    def get_cuts(self) -> Generator[MonoCut, None, None]:
        ds = load_dataset("davidscripka/MIT_environmental_impulse_responses")
        assert isinstance(ds, DatasetDict)

        for data in ds["train"]:
            assert isinstance(data, dict)
            audio = data["audio"]
            assert isinstance(audio, AudioDecoder)
            audio_samples = audio.get_all_samples()
            metadata = audio.metadata
            assert isinstance(metadata, AudioStreamMetadata)

            id = uuid.uuid4().hex
            buf = io.BytesIO()
            sf.write(buf, audio_samples.data.T, metadata.sample_rate, format="WAV")
            recording = Recording.from_bytes(
                buf.getvalue(), recording_id=f"recording_{id}"
            )

            cut = MonoCut(
                id=id,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )

            yield cut
