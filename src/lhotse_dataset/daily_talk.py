import tempfile
import zipfile
from pathlib import Path
from typing import Generator

import gdown
import lhotse

from lhotse_dataset.base import BaseCorpus, Language


class DailyTalk(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://github.com/keonlee9420/DailyTalk"

    @property
    def download_url(self) -> str:
        return "https://drive.google.com/uc?id=1nPrfJn3TcIVPc0Uf5tiAXUYLJceb_5k-"

    @property
    def language(self) -> Language:
        return Language.EN

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "dailytalk.zip"
            gdown.download(self.download_url, str(tmp_path))

            zip_file = zipfile.ZipFile(tmp_path)

            wav_zippaths = [
                file_name.filename
                for file_name in zip_file.filelist
                if file_name.filename.endswith(".wav")
            ]

            for wav_zippath in sorted(wav_zippaths):
                audio_id = Path(wav_zippath).stem
                _, speaker_id, dialogue_id = audio_id.split("_")

                with zip_file.open(wav_zippath, "r") as audio_file:
                    wav_bytes = audio_file.read()
                recording = lhotse.Recording.from_bytes(
                    wav_bytes, f"recording_{audio_id}"
                )

                transcript_zippath = str(Path(wav_zippath).with_suffix(".txt"))
                with zip_file.open(transcript_zippath) as transcript_file:
                    text = transcript_file.read().decode("utf-8")

                supervision = lhotse.SupervisionSegment(
                    id=f"transcript_{audio_id}",
                    recording_id=recording.id,
                    start=0,
                    duration=recording.duration,
                    channel=0,
                    text=text,
                    language=self.language.value,
                    speaker=f"{speaker_id}_{dialogue_id}",
                )
                cut = lhotse.MonoCut(
                    id=f"{audio_id}",
                    start=0,
                    duration=recording.duration,
                    channel=0,
                    supervisions=[supervision],
                    recording=recording,
                )
                yield cut
