import io
import tarfile
import tempfile
from pathlib import Path
from typing import Generator

import git
import lhotse
import numpy as np
import pandas as pd
import soundfile as sf
from lhotse import MultiCut, SupervisionSegment

from lhotse_dataset.base import BaseCorpus, Language
from lhotse_dataset.utils import download_file


class Libri2MixClean(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://github.com/JorisCos/LibriMix"

    @property
    def download_url(self) -> dict[str, str]:
        return {
            "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
            "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        }

    @property
    def language(self) -> Language:
        return Language.EN

    @property
    def shard_size(self) -> int:
        return 5000

    def get_cuts(self) -> Generator[MultiCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir = Path(tmp_dir) / "LibriMix"
            git.Repo.clone_from(self.url, dir)

            libri2mix_csv_paths = dir.glob("**/Libri2Mix/*.csv")

            tmp_dir_path = Path(tmp_dir)

            for csv_path in sorted(list(libri2mix_csv_paths)):
                if str(csv_path).endswith("_info.csv"):
                    continue

                subset = csv_path.stem.split("_")[1]

                df = pd.read_csv(csv_path)

                tmp_ls_path = tmp_dir_path / f"{subset}.tar.gz"
                download_file(self.download_url[subset], tmp_ls_path)

                with tarfile.open(tmp_ls_path) as tar:
                    tar.extractall(tmp_dir_path, filter="fully_trusted")

                for row in df.itertuples():
                    source_1_path = tmp_dir_path / "LibriSpeech" / row.source_1_path  # type: ignore
                    source_2_path = tmp_dir_path / "LibriSpeech" / row.source_2_path  # type: ignore

                    wav_1, sr = sf.read(source_1_path)
                    wav_2, sr = sf.read(source_2_path)
                    wav_len = max(wav_1.shape[0], wav_2.shape[0])

                    wav = np.zeros((2, wav_len), dtype=wav_1.dtype)
                    wav[0, : wav_1.shape[0]] = wav_1 * row.source_1_gain  # type: ignore
                    wav[1, : wav_2.shape[0]] = wav_2 * row.source_2_gain  # type: ignore

                    buf = io.BytesIO()
                    sf.write(buf, wav.T, sr, format="WAV")

                    mixture_id = row.mixture_ID  # type: ignore
                    recording = lhotse.Recording.from_bytes(
                        buf.getvalue(), recording_id=f"recording_{mixture_id}"
                    )
                    assert recording.channel_ids is not None

                    supervision_source_1 = SupervisionSegment(
                        id=f"source_1_{row.source_1_path}",  # type: ignore
                        recording_id=recording.id,
                        start=0,
                        duration=wav_1.shape[0] / sr,
                        channel=0,
                        custom={"wav_len": wav_1.shape[0]},
                    )
                    supervision_source_2 = SupervisionSegment(
                        id=f"source_2_{row.source_2_path}",  # type: ignore
                        recording_id=recording.id,
                        start=0,
                        duration=wav_2.shape[0] / sr,
                        channel=1,
                        custom={"wav_len": wav_2.shape[0]},
                    )

                    cut = MultiCut(
                        id=mixture_id,
                        start=0,
                        duration=recording.duration,
                        supervisions=[supervision_source_1, supervision_source_2],
                        channel=recording.channel_ids,
                        recording=recording,
                        custom={"subset": subset},
                    )

                    yield cut
