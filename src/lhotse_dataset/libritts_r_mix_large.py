import io
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Generator

import lhotse
import numpy as np
import pandas as pd
import soundfile as sf

from lhotse_dataset.base import BaseCorpus, Language
from lhotse_dataset.utils import download_file


class LibriTTSRMixLarge(BaseCorpus):
    @property
    def download_url(self) -> dict[str, str]:
        return {
            "dev-clean": "https://www.openslr.org/resources/141/dev_clean.tar.gz",
            "test-clean": "https://www.openslr.org/resources/141/test_clean.tar.gz",
            "train-clean-100": "https://www.openslr.org/resources/141/train_clean_100.tar.gz",
            "train-clean-360": "https://www.openslr.org/resources/141/train_clean_360.tar.gz",
        }

    @property
    def language(self) -> Language:
        return Language.EN

    @property
    def shard_size(self) -> int:
        return 100

    def get_cuts(self) -> Generator[lhotse.MultiCut, None, None]:
        metadata_dir = Path(__file__).parent / "data/libritts_r_mix_large"
        csv_paths = sorted(list(metadata_dir.glob("*.csv")))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            for csv_path in csv_paths:
                subset = csv_path.stem
                df = pd.read_csv(csv_path)

                tmp_ls_path = tmp_dir_path / f"{subset}.tar.gz"
                download_file(self.download_url[subset], tmp_ls_path)

                with tarfile.open(tmp_ls_path) as tar:
                    tar.extractall(tmp_dir_path, filter="fully_trusted")

                for row in df.itertuples():
                    id_1 = row.id_1  # type: ignore
                    source_1_path = (
                        tmp_dir_path
                        / "LibriTTS_R"
                        / subset
                        / id_1.split("_")[4]
                        / id_1.split("_")[5]
                        / f"{'_'.join(id_1.split('_')[4:])}.wav"
                    )
                    id_2 = row.id_2  # type: ignore
                    source_2_path = (
                        tmp_dir_path
                        / "LibriTTS_R"
                        / subset
                        / id_2.split("_")[4]
                        / id_2.split("_")[5]
                        / f"{'_'.join(id_2.split('_')[4:])}.wav"
                    )

                    wav_1, sr = sf.read(source_1_path)
                    wav_2, sr = sf.read(source_2_path)
                    wav_len = max(wav_1.shape[0], wav_2.shape[0])

                    wav = np.zeros((2, wav_len), dtype=wav_1.dtype)
                    wav[0, : wav_1.shape[0]] = wav_1
                    wav[1, : wav_2.shape[0]] = wav_2

                    buf = io.BytesIO()
                    sf.write(buf, wav.T, sr, format="WAV")

                    mixture_id = uuid.uuid4().hex
                    recording = lhotse.Recording.from_bytes(
                        buf.getvalue(), recording_id=f"recording_{mixture_id}"
                    )
                    assert recording.channel_ids is not None

                    normalized_txt_path_1 = (
                        source_1_path.parent / f"{source_1_path.stem}.normalized.txt"
                    )
                    try:
                        with open(normalized_txt_path_1, "r", encoding="utf-8") as f:
                            normalized_txt_1 = f.readline()
                    except FileNotFoundError:
                        print(
                            "Warning: Normalized text file not found for", source_1_path
                        )
                        normalized_txt_1 = ""

                    original_txt_path_1 = (
                        source_1_path.parent / f"{source_1_path.stem}.original.txt"
                    )
                    try:
                        with open(original_txt_path_1, "r", encoding="utf-8") as f:
                            original_txt_1 = f.readline()
                    except FileNotFoundError:
                        print(
                            "Warning: Original text file not found for", source_1_path
                        )
                        original_txt_1 = ""

                    normalized_txt_path_2 = (
                        source_2_path.parent / f"{source_2_path.stem}.normalized.txt"
                    )
                    try:
                        with open(normalized_txt_path_2, "r", encoding="utf-8") as f:
                            normalized_txt_2 = f.readline()
                    except FileNotFoundError:
                        print(
                            "Warning: Normalized text file not found for", source_2_path
                        )
                        normalized_txt_2 = ""

                    original_txt_path_2 = (
                        source_2_path.parent / f"{source_2_path.stem}.original.txt"
                    )
                    try:
                        with open(original_txt_path_2, "r", encoding="utf-8") as f:
                            original_txt_2 = f.readline()
                    except FileNotFoundError:
                        print(
                            "Warning: Original text file not found for", source_2_path
                        )
                        original_txt_2 = ""

                    supervision_source_1 = lhotse.SupervisionSegment(
                        id=f"source_1_{source_1_path}",
                        recording_id=recording.id,
                        start=0,
                        duration=wav_1.shape[0] / sr,
                        channel=0,
                        text=normalized_txt_1,
                        custom={
                            "wav_len": wav_1.shape[0],
                            "original_txt": original_txt_1,
                        },
                    )
                    supervision_source_2 = lhotse.SupervisionSegment(
                        id=f"source_2_{source_2_path}",
                        recording_id=recording.id,
                        start=0,
                        duration=wav_2.shape[0] / sr,
                        channel=1,
                        text=normalized_txt_2,
                        custom={
                            "wav_len": wav_2.shape[0],
                            "original_txt": original_txt_2,
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

                    yield cut
