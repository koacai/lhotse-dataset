import tempfile
import zipfile
from pathlib import Path
from typing import Generator

import lhotse

from lhotse_dataset.base import BaseCorpus, Gender, Language
from lhotse_dataset.utils import download_file


class JSUT(BaseCorpus):
    @property
    def url(self) -> str:
        return "https://sites.google.com/site/shinnosuketakamichi/publication/jsut"

    @property
    def download_url(self) -> str:
        return "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"

    @property
    def language(self) -> Language:
        return Language.JA

    def get_cuts(self) -> Generator[lhotse.MonoCut, None, None]:
        """Download the corpus to temporary file"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "jsut.zip"
            download_file(self.download_url, tmp_path)

            jsut_zip = zipfile.ZipFile(tmp_path)
            transcript_zippaths = [
                file_name.filename
                for file_name in jsut_zip.filelist
                if "transcript_utf8" in file_name.filename
            ]
            for transcript_zippath in sorted(transcript_zippaths):
                with jsut_zip.open(transcript_zippath) as transcript_file:
                    lines = transcript_file.read().decode("utf-8").splitlines()
                for line in lines:
                    audio_id, text = line.split(":")
                    audio_path = (
                        Path(transcript_zippath).parent / "wav" / f"{audio_id}.wav"
                    )
                    with jsut_zip.open(str(audio_path), "r") as audio_file:
                        wav_bytes = audio_file.read()

                    recording = lhotse.Recording.from_bytes(
                        wav_bytes, f"recording_{audio_id}"
                    )
                    supervision = lhotse.SupervisionSegment(
                        id=f"transcript_{audio_id}",
                        recording_id=recording.id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        text=text,
                        language=self.language.value,
                        speaker=None,
                        gender=Gender.FEMALE.value,
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
