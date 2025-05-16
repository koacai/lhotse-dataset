import tempfile
import zipfile
from pathlib import Path
from typing import Generator

import lhotse

from lhotse_dataset.base import BaseCorpus
from lhotse_dataset.utils import download_file


class WhamNoise(BaseCorpus):
    @property
    def url(self) -> str:
        return "http://wham.whisper.ai/"

    @property
    def download_url(self) -> str:
        return "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"

    @property
    def shard_size(self) -> int:
        return 2000

    def get_cuts(self) -> Generator[lhotse.MultiCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "wham_noise.zip"
            download_file(self.download_url, tmp_path)

            wham_zip = zipfile.ZipFile(tmp_path)
            wav_zippaths = [
                file_name.filename
                for file_name in wham_zip.filelist
                if file_name.filename.endswith(".wav")
            ]

            for wav_zippath in sorted(wav_zippaths):
                audio_id = Path(wav_zippath).stem
                with wham_zip.open(str(wav_zippath), "r") as audio_file:
                    wav_bytes = audio_file.read()

                dir = Path(wav_zippath).parent.name
                if dir == "tr":
                    subset = "train"
                elif dir == "cv":
                    subset = "validation"
                elif dir == "tt":
                    subset = "test"
                else:
                    raise ValueError(f"Unknown subset {dir}")

                recording = lhotse.Recording.from_bytes(
                    wav_bytes, f"recording_{audio_id}"
                )

                cut = lhotse.MultiCut(
                    id=f"{audio_id}",
                    start=0,
                    duration=recording.duration,
                    channel=[0, 1],
                    recording=recording,
                    custom={"subset": subset},
                )
                yield cut
