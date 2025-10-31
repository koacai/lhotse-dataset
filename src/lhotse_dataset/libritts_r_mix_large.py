import io
import uuid
from pathlib import Path
from typing import Generator

import lhotse
import numpy as np
import soundfile as sf
from lhotse import CutSet
from tqdm import tqdm

from lhotse_dataset.base import BaseCorpus, Language


class LibriTTSRMixLarge(BaseCorpus):
    def __init__(self, libritts_r_shar_dir: str | Path) -> None:
        super(LibriTTSRMixLarge, self).__init__()

        if isinstance(libritts_r_shar_dir, str):
            shar_dir = Path(libritts_r_shar_dir)
        else:
            shar_dir = libritts_r_shar_dir
        self.shar_dir = shar_dir

    @property
    def subset_samples(self) -> dict[str, int]:
        return {
            "test_clean": 5000,
            "dev_clean": 5000,
            "train_clean_100": 100000,
            "train_clean_360": 300000,
        }

    @property
    def language(self) -> Language:
        return Language.EN

    @property
    def shard_size(self) -> int:
        return 1000

    def get_cuts(self) -> Generator[lhotse.MultiCut, None, None]:
        cut_paths = sorted(list(map(str, self.shar_dir.glob("cuts.*.jsonl.gz"))))
        recording_paths = sorted(list(map(str, self.shar_dir.glob("recording.*.tar"))))

        cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})
        cuts = cuts.filter(lambda c: c.duration >= 3.0)  # type: ignore
        cuts = cuts.filter(lambda c: c.duration < 20.0)  # type: ignore
        cuts = cuts.filter(lambda c: len(c.supervisions) > 0)  # type: ignore

        for subset, samples in self.subset_samples.items():
            cuts_subset = cuts.filter(
                lambda s: s.supervisions[0].custom["subset"] == subset  # type: ignore
            )
            cuts_subset = cuts_subset.sort_by_duration()

            data_count = 0

            for cut_1 in tqdm(cuts_subset.data, desc=subset):
                if data_count >= samples:
                    break

                for cut_2 in cuts_subset.data:
                    if data_count >= samples:
                        break

                    if cut_1.duration < cut_2.duration:
                        continue
                    if cut_1.supervisions[0].speaker == cut_2.supervisions[0].speaker:
                        continue

                    # Generate mix with probability (adjust as needed)
                    if np.random.uniform(0, 1) < 0.01:
                        wav_1 = cut_1.load_audio()
                        wav_2 = cut_2.load_audio()
                        wav_len = max(wav_1.shape[-1], wav_2.shape[-1])

                        wav = np.zeros((2, wav_len), dtype=wav_1.dtype)
                        wav[0, : wav_1.shape[-1]] = wav_1
                        wav[1, : wav_2.shape[-1]] = wav_2

                        buf = io.BytesIO()
                        sr = cut_1.sampling_rate
                        sf.write(buf, wav.T, sr, format="WAV")

                        mixture_id = uuid.uuid4().hex
                        recording = lhotse.Recording.from_bytes(
                            buf.getvalue(), recording_id=f"recording_{mixture_id}"
                        )
                        assert recording.channel_ids is not None

                        s1 = cut_1.supervisions[0]
                        assert s1.custom is not None
                        supervision_source_1 = lhotse.SupervisionSegment(
                            id=s1.id,
                            recording_id=recording.id,
                            start=0,
                            duration=wav_1.shape[-1] / sr,
                            channel=0,
                            text=s1.text,
                            custom={
                                "wav_len": wav_1.shape[-1],
                                "original_text": s1.custom["original_text"],
                            },
                        )
                        s2 = cut_2.supervisions[0]
                        assert s2.custom is not None
                        supervision_source_2 = lhotse.SupervisionSegment(
                            id=s2.id,
                            recording_id=recording.id,
                            start=0,
                            duration=wav_2.shape[-1] / sr,
                            channel=1,
                            text=s2.text,
                            custom={
                                "wav_len": wav_2.shape[-1],
                                "original_text": s2.custom["original_text"],
                            },
                        )

                        cut = lhotse.MultiCut(
                            id=mixture_id,
                            start=0,
                            duration=recording.duration,
                            supervisions=[supervision_source_1, supervision_source_2],
                            channel=recording.channel_ids,
                            recording=recording,
                            custom={"subset": subset},
                        )
                        data_count += 1
                        yield cut
