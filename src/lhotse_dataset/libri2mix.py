import io
import tempfile
from pathlib import Path
from typing import Callable, Generator

import git
import lhotse
import numpy as np
import pandas as pd
import soundfile as sf
from lhotse import CutSet, MultiCut
from lhotse.cut import Cut

from lhotse_dataset.base import BaseCorpus, Language


class Libri2Mix(BaseCorpus):
    def __init__(self, librispeech_shar_dir: Path) -> None:
        super(Libri2Mix, self).__init__()
        self.librispeech_shar_dir = Path(librispeech_shar_dir)

    @property
    def url(self) -> str:
        return "https://github.com/JorisCos/LibriMix"

    @property
    def language(self) -> Language:
        return Language.EN

    def get_cuts(self) -> Generator[MultiCut, None, None]:
        ls_cut_paths = sorted(
            list(map(str, self.librispeech_shar_dir.glob("cuts.*.jsonl.gz")))
        )
        ls_recording_paths = sorted(
            list(map(str, self.librispeech_shar_dir.glob("recording.*.tar")))
        )
        ls_cuts = CutSet.from_shar(
            {"cuts": ls_cut_paths, "recording": ls_recording_paths}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir = Path(tmp_dir) / "LibriMix"
            git.Repo.clone_from(self.url, dir)

            libri2mix_csv_paths = dir.glob("**/Libri2Mix/*.csv")

            for csv_path in libri2mix_csv_paths:
                if str(csv_path).endswith("_info.csv"):
                    continue

                df = pd.read_csv(csv_path)
                for row in df.itertuples():
                    source_1_id = self.path_to_librispeech_id(row.source_1_path)  # type: ignore
                    source_1_cut = ls_cuts.filter(self._filter_ls_id(source_1_id))[0]

                    source_2_id = self.path_to_librispeech_id(row.source_2_path)  # type: ignore
                    source_2_cut = ls_cuts.filter(self._filter_ls_id(source_2_id))[0]

                    wav_1 = source_1_cut.load_audio() * row.source_1_gain  # type: ignore
                    wav_2 = source_2_cut.load_audio() * row.source_2_gain  # type: ignore

                    wav_len = max(wav_1.shape[1], wav_2.shape[1])
                    wav = np.zeros((2, wav_len), dtype=wav_1.dtype)
                    wav[0, : wav_1.shape[1]] = wav_1
                    wav[1, : wav_2.shape[1]] = wav_2

                    buf = io.BytesIO()
                    sf.write(buf, wav.T, source_1_cut.sampling_rate, format="WAV")

                    mixture_id = row.mixture_ID  # type: ignore
                    recording = lhotse.Recording.from_bytes(
                        buf.getvalue(), recording_id=f"recording_{mixture_id}"
                    )

                    supervision_0 = lhotse.SupervisionSegment(
                        id=f"segment_{mixture_id}_0",
                        recording_id=recording.id,
                        start=0,
                        duration=source_1_cut.supervisions[0].duration,
                        channel=0,
                        text=source_1_cut.supervisions[0].text,
                        language=source_1_cut.supervisions[0].language,
                        speaker=source_1_cut.supervisions[0].speaker,
                        gender=source_1_cut.supervisions[0].gender,
                        custom=source_1_cut.supervisions[0].custom,
                    )
                    supervision_1 = lhotse.SupervisionSegment(
                        id=f"segment_{mixture_id}_1",
                        recording_id=recording.id,
                        start=0,
                        duration=source_2_cut.supervisions[0].duration,
                        channel=1,
                        text=source_2_cut.supervisions[0].text,
                        language=source_2_cut.supervisions[0].language,
                        speaker=source_2_cut.supervisions[0].speaker,
                        gender=source_2_cut.supervisions[0].gender,
                        custom=source_2_cut.supervisions[0].custom,
                    )

                    cut = MultiCut(
                        id=mixture_id,
                        start=0,
                        duration=recording.duration,
                        channel=[0, 1],
                        supervisions=[supervision_0, supervision_1],
                        recording=recording,
                    )
                    yield cut

    @staticmethod
    def path_to_librispeech_id(path: str) -> str:
        subset = path.split("/")[0]
        audio_id = path.split("/")[-1].split(".")[0]
        return f"librispeech_{subset}_{audio_id}"

    @staticmethod
    def _filter_ls_id(id: str) -> Callable[[Cut], bool]:
        return lambda cut: cut.id == id
